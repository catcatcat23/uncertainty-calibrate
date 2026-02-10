"""
Modified baseline_infer_singlepad.py

- Baseline single-model forward to get softmax probabilities:
    NONE_CONF_BASE
    INSUFFICIENT_SOLDER_CONF_BASE
    PSEUDO_SOLDER_CONF_BASE

- Ground truth from CSV soft labels:
    NONE_CONF / INSUFFICIENT_SOLDER_CONF / PSEUDO_SOLDER_CONF (argmax)

- Evaluation:
    * At threshold = 0.5 / 0.8:
        - binary accuracy (defect vs ok)
        - multiclass accuracy (3-class)
        - Binary output: ok/defect precision/recall/F1/omission_rate
        - Multiclass output: ok/undersolder/pseudosolder precision/recall/F1/omission_rate
        - binary F1 curve (defect vs ok)

    * Calibration (ECE + reliability diagrams), **aligned with DeepEnsemble/MC-Dropout/Laplace**:
        Binary:
            - overall ECE on top-1 prediction
            - classwise ECE by TRUE class (true=ok / true=defect),
              using P(ok) / P(defect) as confidences
              -> 3 reliability plots: overall + true=ok + true=defect
        Multiclass:
            - overall ECE on top-1 prediction
            - classwise ECE by TRUE class for ok/undersolder/pseudosolder,
              using the probability of the true class as confidence
              -> 4 reliability plots: overall + ok + undersolder + pseudosolder
            - additional "vector ECE" on full softmax vector (all classes),
              including:
                - vector-ECE and vector-MCE
                - per-bin total gap diagram
                - per-class vector reliability diagrams
"""

import os
import glob
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils.utilities import TransformImage
from models.MPB3 import MPB3net

# ==== 可视化 & F1 曲线 ====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================
# 基本配置
# ==============================

CKPT_DIR = "/home/cat/workspace/vlm/scripts/models/checkpoints"

N_CLASS = 3
N_CLASSES = 3

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROOT_DIR = (
    "/home/cat/workspace/defect_data/"
    "defect_DA758_black_uuid_250310/send2terminal/250310"
)
CSV_PATH = os.path.join(ROOT_DIR, "checked_samples.csv")

CLASS_ORDER = ["ok", "undersolder", "pseudosolder"]

PART_CONFIG = {
    "singlepad": {
        "ckpt_pattern": "singlepadfcdropoutmobilenetv3largers6464*dual2*.pth.tar",
        "backbone": "fcdropoutmobilenetv3large",
        "n_units": [256, 256],
        "n_class": N_CLASS,
        "img_h": 64,
        "img_w": 64,
    },
    "singlepinpad": {
        "ckpt_pattern": "singlepinpadmobilenetv3smallrs12832*dual2*.pth.tar",
        "backbone": "mobilenetv3small",
        "n_units": [256, 256],
        "n_class": N_CLASS,
        "img_h": 32,
        "img_w": 128,
    },
}

# F1 曲线与 ECE 的配置（与其他脚本对齐）
N_THRESH = 101             # F1 曲线阈值采样点
THRESH_LIST = [0.5, 0.8]
N_BINS_ECE = 15            # ECE 直方图 bin 数

SAVE_DIR = os.path.join(ROOT_DIR, "baseline_eval")
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"


# ==============================
# 通用工具：加载 ckpt
# ==============================

def load_model_by_values(ckp, model):
    model_state_dict = model.state_dict()
    model_state_dict_keys = list(model_state_dict.keys())
    index = 0
    new_state_dict = {}

    for key, value in ckp.items():
        if index >= len(model_state_dict_keys):
            print(f"[WARN] ckpt 多出来的参数：{key}")
            break

        model_key = model_state_dict_keys[index]

        if key == model_key and model_state_dict[model_key].shape == value.shape:
            new_state_dict[model_key] = value
        elif model_state_dict[model_key].shape == value.shape:
            print(f"ckp 的 {key} 对应 model 的 {model_key}")
            new_state_dict[model_key] = value
        else:
            print(
                f"Key 不匹配: ckpt {key} ({value.shape}) vs model {model_key} "
                f"({model_state_dict[model_key].shape})"
            )

        index += 1

    model.load_state_dict(new_state_dict, strict=True)


def _find_ckpt(pattern: str) -> str:
    full_pattern = os.path.join(CKPT_DIR, pattern)
    files = glob.glob(full_pattern)
    if not files:
        raise FileNotFoundError(f"未找到匹配 ckpt: {full_pattern}")
    if len(files) > 1:
        print(f"[WARN] 匹配到多个 ckpt，只使用第一个：{files[0]}")
    return files[0]


def _build_model_for_part(part_name: str) -> MPB3net:
    if part_name not in PART_CONFIG:
        raise ValueError(f"未知 part_name: {part_name}，请在 PART_CONFIG 里加配置。")

    cfg = PART_CONFIG[part_name]
    ckpt_path = _find_ckpt(cfg["ckpt_pattern"])
    # 如需固定为某个 ckpt，可以在这里覆盖：
    ckpt_path = (
        "/home/cat/workspace/vlm/scripts/models/checkpoints/singlepadfcdropoutmobilenetv3largers6464s42c3val0.1b256_ckp_v0.18.9lhf1certainlut05cp05clean20.0j0.4lr0.025nb256nm256dual2top2.pth.tar"
    )

    print(f"=> 为 {part_name} 加载 CNN 模型: {ckpt_path}")

    model = MPB3net(
        backbone=cfg["backbone"],
        pretrained=False,
        n_class=cfg["n_class"],
        n_units=cfg["n_units"],
        output_form="dual2",
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[WARN] strict=True 加载失败，尝试按 value 对齐: {e}")
        load_model_by_values(state_dict, model)

    model.to(DEVICE)
    model.eval()
    print(f"=> {part_name} 模型已加载到 {DEVICE}")
    return model


# ==============================
# 原始推理：单次 forward
# ==============================

@torch.no_grad()
def infer_singlepad_batch_base(
    csv_path: str,
    root_path: str,
    model: MPB3net,
    part_name: str = "singlepad",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    用单模型原始 forward 对 CSV 中所有样本做推理。

    返回 DataFrame：
        id, ref_path, insp_path
        NONE_CONF_BASE / INSUFFICIENT_SOLDER_CONF_BASE / PSEUDO_SOLDER_CONF_BASE
    """
    cfg = PART_CONFIG[part_name]
    df = pd.read_csv(csv_path)

    if "part_name" in df.columns:
        df = df[df["part_name"] == part_name].reset_index(drop=True)

    results = []
    num_samples = len(df)
    print(f"[*] 一共 {num_samples} 条样本，开始 Baseline 推理 (batch_size={batch_size})")

    with tqdm(total=num_samples, desc=f"{part_name} Baseline", unit="img") as pbar:
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_df = df.iloc[start:end]

            img1_list, img2_list = [], []
            meta_list = []

            for _, row in batch_df.iterrows():
                ref_rel = row["ref_image"]
                insp_rel = row["insp_image"]
                sample_id = row["id"]

                ref_path = os.path.join(root_path, ref_rel)
                insp_path = os.path.join(root_path, insp_rel)

                ref_img = TransformImage(
                    img_path=ref_path,
                    rs_img_size_h=cfg["img_h"],
                    rs_img_size_w=cfg["img_w"],
                ).transform()

                insp_img = TransformImage(
                    img_path=insp_path,
                    rs_img_size_h=cfg["img_h"],
                    rs_img_size_w=cfg["img_w"],
                ).transform()

                ref_tensor = torch.FloatTensor(ref_img)
                insp_tensor = torch.FloatTensor(insp_img)

                img1_list.append(ref_tensor)
                img2_list.append(insp_tensor)

                meta_list.append({
                    "id": sample_id,
                    "ref_path": ref_rel,
                    "insp_path": insp_rel,
                })

            x1 = torch.cat(img1_list, dim=0).to(DEVICE)
            x2 = torch.cat(img2_list, dim=0).to(DEVICE)

            _, logits_bom = model(x1, x2)
            probs = F.softmax(logits_bom, dim=-1)  # [B, 3]
            probs_np = probs.cpu().numpy()

            for i, meta in enumerate(meta_list):
                p = probs_np[i]
                row_result = {
                    **meta,
                    "NONE_CONF_BASE":                float(p[0]),
                    "INSUFFICIENT_SOLDER_CONF_BASE": float(p[1]),
                    "PSEUDO_SOLDER_CONF_BASE":       float(p[2]),
                }
                results.append(row_result)

            pbar.update(len(batch_df))

    result_df = pd.DataFrame(results)
    cols = [
        "id",
        "ref_path", "insp_path",
        "NONE_CONF_BASE",
        "INSUFFICIENT_SOLDER_CONF_BASE",
        "PSEUDO_SOLDER_CONF_BASE",
    ]
    result_df = result_df[cols]
    return result_df


# ==============================
# 评估工具：混淆矩阵 & F1
# ==============================

def confusion_matrix_3cls(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_class: int = N_CLASSES,
):
    cm = np.zeros((n_class, n_class), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def calc_precision_recall_f1_binary(
    y_true_bin: np.ndarray,
    scores_pos: np.ndarray,
    thresholds: np.ndarray,
):
    precisions, recalls, f1s = [], [], []

    for thr in thresholds:
        y_pred_bin = (scores_pos >= thr).astype(np.int64)

        tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
        fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
        fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())

        if tp + fp == 0:
            prec = 1.0
        else:
            prec = tp / (tp + fp)

        if tp + fn == 0:
            rec = 0.0
        else:
            rec = tp / (tp + fn)

        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return (
        np.array(precisions, dtype=np.float32),
        np.array(recalls, dtype=np.float32),
        np.array(f1s, dtype=np.float32),
        thresholds,
    )


def metrics_from_cm(cm: np.ndarray):
    C = cm.shape[0]
    per_prec = np.zeros(C, dtype=float)
    per_rec  = np.zeros(C, dtype=float)
    per_f1   = np.zeros(C, dtype=float)
    supports = cm.sum(axis=1)
    total_support = supports.sum()

    weighted_f1_num = 0.0

    for c in range(C):
        tp = cm[c, c]
        support = supports[c]
        pred_pos = cm[:, c].sum()

        if support == 0:
            rec = float("nan")
        else:
            rec = tp / support

        if pred_pos == 0:
            prec = float("nan")
        else:
            prec = tp / pred_pos

        if np.isnan(prec) or np.isnan(rec) or prec + rec == 0:
            f1 = float("nan")
        else:
            f1 = 2 * prec * rec / (prec + rec)

        per_prec[c], per_rec[c], per_f1[c] = prec, rec, f1

        if not np.isnan(f1):
            weighted_f1_num += f1 * support

    if total_support > 0:
        weighted_f1 = weighted_f1_num / total_support
    else:
        weighted_f1 = float("nan")

    return per_prec, per_rec, per_f1, supports, weighted_f1


# ==============================
# ECE & 可靠性图工具（标量事件版）
# ==============================

def compute_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = N_BINS_ECE,
):
    """
    通用 ECE 计算（与其他脚本统一实现）：
        confidences: [N] 模型给“某个事件”的概率 (0~1)
        correctness: [N] 0/1，该事件是否发生
    返回：
        ece, bin_edges, bin_conf, bin_acc, bin_counts
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)

    conf = np.clip(conf, 0.0, 1.0)
    assert conf.shape == corr.shape

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = np.zeros(n_bins, dtype=float)
    bin_acc = np.zeros(n_bins, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    # digitize: 返回 bin index (0..n_bins-1)
    bin_ids = np.digitize(conf, bin_edges[1:-1], right=True)

    for b in range(n_bins):
        mask = (bin_ids == b)
        cnt = int(mask.sum())
        bin_counts[b] = cnt
        if cnt > 0:
            bin_conf[b] = float(conf[mask].mean())
            bin_acc[b] = float(corr[mask].mean())
        else:
            bin_conf[b] = 0.0
            bin_acc[b] = 0.0

    N = conf.shape[0]
    ece = 0.0
    for b in range(n_bins):
        if bin_counts[b] == 0:
            continue
        weight = bin_counts[b] / N
        ece += weight * abs(bin_acc[b] - bin_conf[b])

    return ece, bin_edges, bin_conf, bin_acc, bin_counts


def plot_reliability_bars(
    bin_edges: np.ndarray,
    bin_acc: np.ndarray,
    title: str,
    out_path: str,
):
    """
    画柱状可靠性图：每个 bin 一个柱子表示经验 accuracy，
    并叠加 y=x 的虚线表示完美校准。
    """
    n_bins = len(bin_acc)
    width = bin_edges[1] - bin_edges[0]
    centers = bin_edges[:-1] + width / 2.0

    plt.figure(figsize=(5, 5))

    plt.bar(
        centers,
        bin_acc,
        width=width * 0.9,
        align="center",
        alpha=0.7,
        edgecolor="black",
        label="Empirical accuracy",
    )
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] reliability plot: {out_path}")


# ==============================
# Multiclass “向量版” ECE 及绘图
# ==============================

def compute_multiclass_vector_ece(
    softmaxes: np.ndarray,
    labels: np.ndarray,
    n_bins: int = N_BINS_ECE,
):
    """
    Multiclass 向量版 ECE（按 max 置信度分 bin + 按预测类别细分）:

    对每个样本 i:
      - probs[i, :] 为 softmax 概率
      - pred[i] = argmax_k probs[i, k] 为预测类别
      - max_conf[i] = probs[i, pred[i]] 为最大置信度

    步骤:
      1) 按 max_conf 分成若干个置信度区间（bin）。
      2) 在每个 bin 中，再按预测类别 pred 分成 C 组。
      3) 对每个 (bin b, 类别 k):
           acc_{b,k}  = 该组中 true label == k 的比例
           conf_{b,k} = 该组中 probs[:, k] 的平均值 (等同于 max_conf)
           gap_{b,k}  = |acc_{b,k} - conf_{b,k}|
      4) 全局 ECE:
           ECE = Σ_{b,k} (|S_{b,k}| / N) * gap_{b,k}
         其中 S_{b,k} 是 bin b 中预测为 k 的样本集合。
      5) per-bin gap:
           g_b = Σ_k (|S_{b,k}| / |S_b|) * gap_{b,k}
         MCE 取 max_b g_b。

    返回:
        ece:        标量，全局 ECE
        mce:        标量，最大 per-bin gap
        bin_edges:  [n_bins+1]，bin 边界
        bin_counts: [n_bins]，每个 bin 的样本数 |S_b|
        gap_per_bin:[n_bins]，每个 bin 的 g_b
        acc_means:  [C, n_bins]，acc_{b,k}（没样本时为 NaN）
        conf_means: [C, n_bins]，conf_{b,k}（没样本时为 NaN）
    """
    probs = np.asarray(softmaxes, dtype=np.float64)   # [N, C]
    labels = np.asarray(labels, dtype=np.int64)       # [N]
    N, C = probs.shape

    # top-1 预测类别 & 对应最大置信度
    pred = probs.argmax(axis=1)                       # [N]
    max_conf = probs[np.arange(N), pred]             # [N]

    # 按 max_conf 等宽分 bin
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    # 结果数组
    acc_means = np.full((C, n_bins), np.nan, dtype=np.float64)
    conf_means = np.full((C, n_bins), np.nan, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=int)
    gap_per_bin = np.zeros(n_bins, dtype=np.float64)

    ece = 0.0
    mce = 0.0

    for b, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # 当前 bin 内的样本
        in_bin = (max_conf > lower) & (max_conf <= upper)
        count_b = int(in_bin.sum())
        bin_counts[b] = count_b
        if count_b == 0:
            continue

        # 该 bin 内的“局部 ECE”
        g_b = 0.0

        # 在 bin 内再按预测类别分组
        for k in range(C):
            in_bin_k = in_bin & (pred == k)
            count_bk = int(in_bin_k.sum())
            if count_bk == 0:
                continue

            # acc_{b,k}：这一组里预测对的比例
            acc_k = (labels[in_bin_k] == k).mean()

            # conf_{b,k}：这一组里对类别 k 的平均置信度
            # 因为 pred == k，这里用 probs[:, k] 或 max_conf 都一样
            conf_k = probs[in_bin_k, k].mean()

            acc_means[k, b] = acc_k
            conf_means[k, b] = conf_k

            gap = abs(acc_k - conf_k)

            # 组内权重：该组在当前 bin 内的占比
            weight_in_bin = count_bk / count_b
            g_b += weight_in_bin * gap

            # 全局 ECE 的权重：该组在全体样本中的占比
            ece += (count_bk / N) * gap

        gap_per_bin[b] = g_b
        if g_b > mce:
            mce = g_b

    return ece, mce, bin_edges, bin_counts, gap_per_bin, acc_means, conf_means


def plot_multiclass_vector_gap(
    bin_edges: np.ndarray,
    gap_per_bin: np.ndarray,
    title: str,
    out_path: str,
):
    """
    画每个 bin 的向量 gap g_m 的柱形图。
    """
    width = bin_edges[1] - bin_edges[0]
    centers = bin_edges[:-1] + width / 2.0

    plt.figure(figsize=(6, 4))
    plt.bar(
        centers,
        gap_per_bin,
        width=width * 0.9,
        align="center",
        alpha=0.7,
        edgecolor="black",
        label=r"$\sum_k |acc_k - conf_k|$",
    )

    plt.xlabel("Max predicted probability bin")
    plt.ylabel("Per-bin vector gap")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] multiclass vector gap diagram saved to: {out_path}")


def plot_multiclass_vector_classwise(
    bin_edges: np.ndarray,
    acc_means: np.ndarray,
    conf_means: np.ndarray,
    class_idx: int,
    class_name: str,
    title_prefix: str,
    out_path: str,
):
    """
    对某一个类别画“向量版 per-class gap 图”：
      - x 轴：max_conf 的 bin 中心
      - 柱形：该 bin 内该类的 gap = |acc_k - conf_k|
        （acc_k: 该 bin 内 true=class_k 的频率；
         conf_k: 该 bin 内该类预测概率的平均值）
    """
    acc = acc_means[class_idx]    # [n_bins]
    conf = conf_means[class_idx]  # [n_bins]
    gap = np.abs(acc - conf)      # [n_bins]

    n_bins = len(gap)
    width = bin_edges[1] - bin_edges[0]
    centers = bin_edges[:-1] + width / 2.0

    # 没有样本的 bin 会是 NaN，过滤掉
    mask = ~np.isnan(gap)

    plt.figure(figsize=(5, 5))
    plt.bar(
        centers[mask],
        gap[mask],
        width=width * 0.9,
        align="center",
        alpha=0.7,
        edgecolor="black",
        label=r"|acc - conf| (class=%s)" % class_name,
    )

    plt.xlabel("Max predicted probability bin")
    plt.ylabel(r"Per-bin gap |acc - conf|")
    plt.title(f"{title_prefix} (class={class_name})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)  # 对于单类 gap, 取值范围 [0,1]
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(
        f"[INFO] multiclass vector per-class gap "
        f"(class={class_name}) saved to: {out_path}"
    )



# ==============================
# 评估主函数：分类指标 + ECE + 可靠性图 + 向量 ECE
# ==============================

def eval_singlepad_baseline(
    csv_path: str,
    save_dir: str,
    part_name: str = "singlepad",
):
    df = pd.read_csv(csv_path)

    if "part_name" in df.columns:
        df = df[df["part_name"] == part_name].reset_index(drop=True)

    if len(df) == 0:
        print(f"[WARN] CSV 中没有 part_name={part_name} 的数据，跳过评估")
        return

    # ground truth：来自原 soft label 列
    gt_cols = ["NONE_CONF", "INSUFFICIENT_SOLDER_CONF", "PSEUDO_SOLDER_CONF"]
    if not all(col in df.columns for col in gt_cols):
        raise ValueError(
            "CSV 缺少 ground truth 列："
            "NONE_CONF / INSUFFICIENT_SOLDER_CONF / PSEUDO_SOLDER_CONF"
        )

    methods = {
        "Baseline": {
            "cols": [
                "NONE_CONF_BASE",
                "INSUFFICIENT_SOLDER_CONF_BASE",
                "PSEUDO_SOLDER_CONF_BASE",
            ],
            "color": "C3",
            "marker": "x",
        },
    }

    thresholds_curve = np.linspace(0.0, 1.0, N_THRESH)
    f1_curves: Dict[str, Dict[str, np.ndarray]] = {}

    soft_gt = df[gt_cols].to_numpy(dtype=np.float32)
    y_true_3 = soft_gt.argmax(axis=1).astype(np.int64)
    y_true_bin = (y_true_3 != 0).astype(np.int64)  # defect = 非 ok

    print("\n===== singlepad Baseline evaluation (3-class & binary + calibration) =====")

    for method_name, cfg in methods.items():
        prob_cols = cfg["cols"]
        if not all(col in df.columns for col in prob_cols):
            print(f"[WARN] CSV 缺少 {method_name} 概率列，跳过: {prob_cols}")
            continue

        probs = df[prob_cols].to_numpy(dtype=np.float32)  # [N, 3]

        # ---------- Multiclass 分类指标 ----------
        y_pred_3 = probs.argmax(axis=1)
        acc_mclass = (y_pred_3 == y_true_3).mean()

        cm_multi = confusion_matrix_3cls(y_true_3, y_pred_3, n_class=3)
        (
            per_prec_multi,
            per_rec_multi,
            per_f1_multi,
            support_multi,
            weighted_f1_multi,
        ) = metrics_from_cm(cm_multi)

        # ---------- Binary 分类指标 ----------
        p_ok = probs[:, 0]
        p_defect = 1.0 - p_ok
        scores_defect = p_defect  # P(defect)

        _, _, f1s_curve, _ = calc_precision_recall_f1_binary(
            y_true_bin, scores_defect, thresholds_curve
        )
        f1_curves[method_name] = {
            "f1": f1s_curve,
            "color": cfg["color"],
            "marker": cfg["marker"],
        }

        for thr in THRESH_LIST:
            print(f"\nThreshold = {thr:.2f}  --  [{method_name}]")
            print(f"---- threshold = {thr:.2f} ----")

            y_pred_bin = (scores_defect >= thr).astype(np.int64)
            acc_bin = (y_pred_bin == y_true_bin).mean()

            cm_bin = confusion_matrix_3cls(y_true_bin, y_pred_bin, n_class=2)
            (
                per_prec_bin,
                per_rec_bin,
                per_f1_bin,
                support_bin,
                weighted_f1_bin,
            ) = metrics_from_cm(cm_bin)

            print(f"binary accuracy = {acc_bin:.5f}")
            print(f"mclass  accuracy = {acc_mclass:.5f}")

            print("== Binary output ==")
            print(f"weighted f1 score = {weighted_f1_bin}")

            bin_names = ["ok", "defect"]
            for c in range(2):
                name = bin_names[c]
                n_c = int(support_bin[c])
                prec_c = per_prec_bin[c]
                rec_c = per_rec_bin[c]
                f1_c = per_f1_bin[c]

                if np.isnan(rec_c):
                    omission = 0.0
                else:
                    omission = 1.0 - rec_c

                print(
                    f"{name} (n={n_c}): "
                    f"precision:{0.0 if np.isnan(prec_c) else prec_c:.3f}, "
                    f"recall:{0.0 if np.isnan(rec_c) else rec_c:.3f}, "
                    f"f1_score:{0.0 if np.isnan(f1_c) else f1_c:.3f}, "
                    f"omission_rate:{omission:.3f}"
                )

            print("== Multiclass output ==")
            print(f"weighted f1 score = {weighted_f1_multi}")

            for c in range(N_CLASSES):
                name = CLASS_ORDER[c]
                n_c = int(support_multi[c])
                prec_c = per_prec_multi[c]
                rec_c = per_rec_multi[c]
                f1_c = per_f1_multi[c]

                if np.isnan(rec_c):
                    omission = 0.0
                else:
                    omission = 1.0 - rec_c

                print(
                    f"{name} (n={n_c}): "
                    f"precision:{0.0 if np.isnan(prec_c) else prec_c:.3f}, "
                    f"recall:{0.0 if np.isnan(rec_c) else rec_c:.3f}, "
                    f"f1_score:{0.0 if np.isnan(f1_c) else f1_c:.3f}, "
                    f"omission_rate:{omission:.3f}"
                )

        # ---------- Binary ECE & 可靠性（与 deepensemble/mcdropout 对齐） ----------
        # top-1 binary 预测（0.5 阈值）及其置信度
        y_pred_bin_ece = (p_defect >= 0.5).astype(np.int64)
        conf_bin_pred = np.where(y_pred_bin_ece == 1, p_defect, p_ok)
        correct_bin = (y_pred_bin_ece == y_true_bin).astype(np.int64)

        # overall ECE
        ece_bin_overall, bin_edges_b, _, bin_acc_b, _ = compute_ece(
            conf_bin_pred, correct_bin, n_bins=N_BINS_ECE
        )

        print(f"\n[{method_name}] Binary calibration (pred-wise top1)")
        print(f"  overall ECE = {ece_bin_overall:.4f}")

        # overall reliability
        bin_overall_path = os.path.join(
            save_dir, f"singlepad_binary_reliability_overall_{method_name}.png"
        )
        plot_reliability_bars(
            bin_edges_b,
            bin_acc_b,
            title=f"Binary reliability overall (top1, {method_name})",
            out_path=bin_overall_path,
        )

        # TRUE-wise classwise ECE：true=ok / true=defect
        conf_ok = p_ok
        corr_ok = (y_true_bin == 0).astype(np.int64)
        ece_ok, bin_edges_ok, _, bin_acc_ok, _ = compute_ece(
            conf_ok, corr_ok, n_bins=N_BINS_ECE
        )
        path_ok = os.path.join(
            save_dir,
            f"singlepad_binary_reliability_true_ok_{method_name}.png",
        )
        plot_reliability_bars(
            bin_edges_ok,
            bin_acc_ok,
            title=f"Binary reliability (true=ok, {method_name})",
            out_path=path_ok,
        )

        conf_def = p_defect
        corr_def = (y_true_bin == 1).astype(np.int64)
        ece_def, bin_edges_def, _, bin_acc_def, _ = compute_ece(
            conf_def, corr_def, n_bins=N_BINS_ECE
        )
        path_def = os.path.join(
            save_dir,
            f"singlepad_binary_reliability_true_defect_{method_name}.png",
        )
        plot_reliability_bars(
            bin_edges_def,
            bin_acc_def,
            title=f"Binary reliability (true=defect, {method_name})",
            out_path=path_def,
        )

        print(
            "  classwise ECE (prob-wise, by true class): "
            f"ok={ece_ok:.4f}, defect={ece_def:.4f}"
        )

        # ---------- Multiclass ECE & 可靠性（top1 + true-class prob-wise） ----------
        conf_multi_pred = probs[np.arange(len(probs)), y_pred_3]
        correct_multi = (y_pred_3 == y_true_3).astype(np.int64)

        # overall top1
        ece_multi_overall, bin_edges_m, _, bin_acc_m, _ = compute_ece(
            conf_multi_pred, correct_multi, n_bins=N_BINS_ECE
        )

        print(f"\n[{method_name}] Multiclass calibration (top1 + true-class prob-wise)")
        print(f"  overall ECE (top1) = {ece_multi_overall:.4f}")

        multi_overall_path = os.path.join(
            save_dir, f"singlepad_multiclass_reliability_overall_{method_name}.png"
        )
        plot_reliability_bars(
            bin_edges_m,
            bin_acc_m,
            title=f"Multiclass reliability overall (top1, {method_name})",
            out_path=multi_overall_path,
        )

        # true-class-wise ECE：对每一类 c，用 probs[:, c] vs (y_true_3 == c)
        ece_multi_class = {}
        for c, cname in enumerate(CLASS_ORDER):
            conf_c = probs[:, c]
            corr_c = (y_true_3 == c).astype(np.int64)

            ece_c, bin_edges_c, _, bin_acc_c, _ = compute_ece(
                conf_c, corr_c, n_bins=N_BINS_ECE
            )
            ece_multi_class[c] = ece_c

            path_c = os.path.join(
                save_dir,
                f"singlepad_multiclass_reliability_true_{cname}_{method_name}.png",
            )
            plot_reliability_bars(
                bin_edges_c,
                bin_acc_c,
                title=f"Multiclass reliability (true={cname}, {method_name})",
                out_path=path_c,
            )

        print(
            "  classwise ECE (prob-wise, by true class): "
            f"ok={ece_multi_class.get(0, float('nan')):.4f}, "
            f"undersolder={ece_multi_class.get(1, float('nan')):.4f}, "
            f"pseudosolder={ece_multi_class.get(2, float('nan')):.4f}"
        )

        # ---------- Multiclass “向量版” ECE ----------
        (
            ece_vec,
            mce_vec,
            bin_edges_vec,
            bin_counts_vec,
            gap_per_bin,
            acc_means_vec,
            conf_means_vec,
        ) = compute_multiclass_vector_ece(
            probs, y_true_3, n_bins=N_BINS_ECE
        )

        print(f"\n[{method_name}] Multiclass vector calibration (all-class probs)")
        print(f"  vector-ECE = {ece_vec:.4f}")
        print(f"  vector-MCE = {mce_vec:.4f}")

        gap_path = os.path.join(
            save_dir,
            f"singlepad_multiclass_vector_gap_{method_name}.png",
        )
        plot_multiclass_vector_gap(
            bin_edges_vec,
            gap_per_bin,
            title=f"Multiclass vector gap per bin ({method_name})",
            out_path=gap_path,
        )

        for c, cname in enumerate(CLASS_ORDER):
            out_c = os.path.join(
                save_dir,
                f"singlepad_multiclass_vector_reliability_{cname}_{method_name}.png",
            )
            plot_multiclass_vector_classwise(
                bin_edges_vec,
                acc_means_vec,
                conf_means_vec,
                class_idx=c,
                class_name=cname,
                title_prefix="Multiclass vector reliability",
                out_path=out_c,
            )

    # ---------- 画 binary F1 曲线 ----------
    if not f1_curves:
        print("[WARN] 没有方法产生 F1 曲线，跳过画图。")
        return

    plt.figure(figsize=(8, 5))
    for method_name, info in f1_curves.items():
        f1s = info["f1"]
        color = info["color"]

        best_idx = int(f1s.argmax())
        best_thr = thresholds_curve[best_idx]
        best_f1 = f1s[best_idx]

        print(f"\n[{method_name}] F1 curve (binary defect vs ok)")
        print(f"  Max F1 = {best_f1:.4f} at threshold = {best_thr:.3f}")

        plt.plot(
            thresholds_curve,
            f1s,
            label=f"{method_name} (max F1={best_f1:.3f} @ {best_thr:.2f})",
            color=color,
        )

    plt.xlabel("Threshold on P(defect)")
    plt.ylabel("F1 score (defect vs ok)")
    plt.title("singlepad F1 curve vs threshold (Baseline)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best")
    out_f1 = os.path.join(save_dir, "singlepad_f1_curve_defect_vs_ok_Baseline.png")
    plt.tight_layout()
    plt.savefig(out_f1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] F1 曲线图: {out_f1}")


# ==============================
# main：推理 + 写回 CSV + 评估
# ==============================

if __name__ == "__main__":
    part_name = "singlepad"

    # 1. 加载模型
    model = _build_model_for_part(part_name)

    # 2. Baseline 推理
    result_df = infer_singlepad_batch_base(
        csv_path=CSV_PATH,
        root_path=ROOT_DIR,
        model=model,
        part_name=part_name,
        batch_size=64,
    )
    print(f"result df length: {len(result_df)}")

    # 3. 按 id 写回 CSV
    orig_df = pd.read_csv(CSV_PATH)
    print(f"origin df length: {len(orig_df)}")

    dup_id = orig_df["id"].duplicated().sum()
    if dup_id > 0:
        raise ValueError(f"id 列出现重复: {dup_id} 行，请先清洗再推理")

    orig_df = orig_df.set_index("id")
    result_df = result_df.set_index("id")

    cols = [
        "NONE_CONF_BASE",
        "INSUFFICIENT_SOLDER_CONF_BASE",
        "PSEUDO_SOLDER_CONF_BASE",
    ]
    orig_df[cols] = result_df[cols]

    merged = orig_df.reset_index()
    merged.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print("Baseline 概率结果已按 id 回写到:", CSV_PATH)

    # 4. 评估：分类指标 + F1 曲线 + ECE + 可靠性图 + 向量 ECE
    eval_singlepad_baseline(
        csv_path=CSV_PATH,
        save_dir=SAVE_DIR,
        part_name=part_name,
    )
