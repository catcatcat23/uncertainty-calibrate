import os
import glob
import re
import torch.serialization
from typing import Dict, List, Optional

# ==========================
# 关键修复：添加所有需要的安全全局对象
# ==========================
from torch.optim.swa_utils import SWALR  # 导入SWALR
torch.serialization.add_safe_globals([getattr, SWALR])  # 同时添加getattr和SWALR到安全列表

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.optim.swa_utils import AveragedModel

# ==== 可视化 & F1 曲线 ====
import matplotlib
matplotlib.use("Agg")  # 无GUI环境下绘图
import matplotlib.pyplot as plt

# 导入项目内的工具和模型（确保路径正确）
from utils.utilities import TransformImage
from models.MPB3 import MPB3net

# ==========================
# 核心：兼容式模型加载函数（已修复weights_only问题）
# ==========================
def load_mpb3_ckpt_auto(
    ckpt_path: str,
    backbone_arch: str,
    n_class: int,
    n_units: List[int],
    output_type: str = "dual2",
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    自动适配加载 MPB3net 模型检查点，兼容：
      - 普通单卡/多卡训练的模型 (key: xxx / module.xxx)
      - SWA AveragedModel 模型 (key: n_averaged + module.xxx)
      - 各种变体 (module.module.xxx / model.xxx 等)
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _extract_state_dict(obj):
        if not isinstance(obj, dict):
            return obj
        cand_keys = [
            "swa_state_dict",
            "swa_model_state_dict",
            "averaged_model_state_dict",
            "avg_state_dict",
            "state_dict",
            "model_state_dict",
            "model",
            "net",
        ]
        for k in cand_keys:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj

    def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        has = any(k.startswith(prefix) for k in sd.keys())
        if not has:
            return sd
        out = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
            else:
                out[k] = v
        return out

    def _gen_variants(sd: Dict[str, torch.Tensor]):
        variants = []
        seen = set()

        def _add(d):
            keys = tuple(list(d.keys())[:50])
            if keys in seen:
                return
            seen.add(keys)
            variants.append(d)

        _add(sd)
        sd1 = _strip_prefix(sd, "module.")
        _add(sd1)
        sd2 = _strip_prefix(sd1, "module.")
        _add(sd2)
        sd3 = _strip_prefix(sd, "model.")
        _add(sd3)
        sd4 = _strip_prefix(sd3, "module.")
        _add(sd4)
        return variants

    # 安全加载（已添加getattr和SWALR到安全列表）
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict_raw = _extract_state_dict(ckpt)

    # 初始化基础模型
    base_model = MPB3net(
        backbone=backbone_arch,
        pretrained=False,
        n_class=n_class,
        n_units=n_units,
        output_form=output_type,
    )

    fname = os.path.basename(ckpt_path).lower()
    is_swa = ("swa" in fname) or fname.endswith("swa.pth.tar")

    if is_swa:
        print(f"=> [SWA] loading via AveragedModel: {ckpt_path}")
        swa_model = AveragedModel(base_model)

        loaded = False
        variants = _gen_variants(state_dict_raw)

        # 优先尝试：直接加载到SWA模型（strict=True）
        for sd in variants:
            try:
                swa_model.load_state_dict(sd, strict=True)
                loaded = True
                print("[SWA] loaded as AveragedModel.state_dict() (strict=True).")
                break
            except Exception:
                pass

        # 尝试strict=False加载到SWA模型
        if not loaded:
            for sd in variants:
                try:
                    missing, unexpected = swa_model.load_state_dict(sd, strict=False)
                    expected = set(swa_model.state_dict().keys())
                    common = len(expected.intersection(set(sd.keys())))
                    ratio = common / max(1, len(expected))
                    if ratio >= 0.60:
                        loaded = True
                        print(f"[SWA] loaded as AveragedModel (strict=False), "
                              f"common_ratio={ratio:.2f}, missing={len(missing)}, unexpected={len(unexpected)}")
                        break
                except Exception:
                    pass

        # 尝试加载到SWA模型的module（当作普通模型）
        if not loaded:
            for sd in variants:
                try:
                    missing, unexpected = swa_model.module.load_state_dict(sd, strict=True)
                    loaded = True
                    print("[SWA] ckpt loaded into swa_model.module (strict=True).")
                    break
                except Exception:
                    pass

        # 最后尝试strict=False加载到SWA模型的module
        if not loaded:
            for sd in variants:
                try:
                    missing, unexpected = swa_model.module.load_state_dict(sd, strict=False)
                    expected = set(swa_model.module.state_dict().keys())
                    common = len(expected.intersection(set(sd.keys())))
                    ratio = common / max(1, len(expected))
                    if ratio >= 0.60:
                        loaded = True
                        print(f"[SWA] ckpt loaded into swa_model.module (strict=False), "
                              f"common_ratio={ratio:.2f}, missing={len(missing)}, unexpected={len(unexpected)}")
                        break
                except Exception:
                    pass

        if not loaded:
            first_keys = list(state_dict_raw.keys())[:30] if isinstance(state_dict_raw, dict) else []
            raise RuntimeError(
                "[SWA] 无法加载该 ckpt 到 AveragedModel 或其 module。\n"
                f"ckpt_path={ckpt_path}\n"
                f"first_keys={first_keys}\n"
                "请检查 backbone/n_units/n_class/output_type 是否与训练一致。"
            )

        # 补全SWA模型的n_averaged属性（避免推理时属性缺失）
        try:
            if not hasattr(swa_model, "n_averaged") or swa_model.n_averaged is None:
                swa_model.n_averaged = torch.tensor(1, dtype=torch.long)
            else:
                swa_model.n_averaged.copy_(torch.tensor(1, dtype=torch.long))
        except Exception:
            pass

        model = swa_model
        # 若后续代码依赖MPB3net的原生属性（如cnn_encoder），取消下面注释：
        # model = swa_model.module

    else:
        # 非SWA模型加载逻辑
        print(f"=> loading normal ckpt: {ckpt_path}")
        loaded = False
        variants = _gen_variants(state_dict_raw)

        # 尝试strict=True加载
        for sd in variants:
            try:
                base_model.load_state_dict(sd, strict=True)
                loaded = True
                break
            except Exception:
                pass

        # 尝试strict=False加载（兼容少量参数不匹配）
        if not loaded:
            for sd in variants:
                try:
                    missing, unexpected = base_model.load_state_dict(sd, strict=False)
                    expected = set(base_model.state_dict().keys())
                    common = len(expected.intersection(set(sd.keys())))
                    ratio = common / max(1, len(expected))
                    if ratio >= 0.60:
                        loaded = True
                        print(f"[WARN] normal ckpt loaded with strict=False, common_ratio={ratio:.2f}, "
                              f"missing={len(missing)}, unexpected={len(unexpected)}")
                        break
                except Exception:
                    pass

        if not loaded:
            first_keys = list(state_dict_raw.keys())[:30] if isinstance(state_dict_raw, dict) else []
            raise RuntimeError(
                "[NORMAL] 无法加载该 ckpt 到 base_model。\n"
                f"ckpt_path={ckpt_path}\n"
                f"first_keys={first_keys}\n"
                "请检查 backbone/n_units/n_class/output_type 是否与训练一致。"
            )

        model = base_model

    # 模型移至指定设备并设为eval模式（推理必备）
    model = model.to(device).eval()
    return model

# ==========================
# 基本配置
# ==========================
CKPT_DIR = "/home/cat/workspace/vlm/scripts/models/checkpoints/p2"

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

N_THRESH = 101
THRESH_LIST = [0.5, 0.8]
N_BINS_ECE = 15

SAVE_DIR = os.path.join(ROOT_DIR, "deepensemble_eval")
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"

# ==========================
# 通用工具函数
# ==========================
def _find_ckpts(pattern: str, max_models: Optional[int] = None) -> List[str]:
    full_pattern = os.path.join(CKPT_DIR, pattern)
    files = glob.glob(full_pattern)
    if not files:
        raise FileNotFoundError(f"未找到匹配 ckpt: {full_pattern}")

    files = sorted(files)
    if max_models is not None:
        files = files[-max_models:]

    print(f"[INFO] 一共找到 {len(files)} 个 ckpt 用于 deep ensemble：")
    for f in files:
        print("   -", f)
    return files

def _build_ensemble_for_part(
    part_name: str,
    max_models: Optional[int] = None,
) -> List[MPB3net]:
    if part_name not in PART_CONFIG:
        raise ValueError(f"未知 part_name: {part_name}")

    cfg = PART_CONFIG[part_name]
    ckpt_paths = _find_ckpts(cfg["ckpt_pattern"], max_models=max_models)

    models: List[MPB3net] = []
    for ckpt_path in ckpt_paths:
        print(f"=> 为 {part_name} 加载一个 ensemble 成员: {ckpt_path}")
        model = load_mpb3_ckpt_auto(
            ckpt_path=ckpt_path,
            backbone_arch=cfg["backbone"],
            n_class=cfg["n_class"],
            n_units=cfg["n_units"],
            output_type="dual2",
            device=DEVICE
        )
        models.append(model)

    print(f"=> {part_name} 的 deep ensemble 共 {len(models)} 个模型，已加载到 {DEVICE}")
    return models

@torch.no_grad()  # 推理时禁用梯度计算（节省显存+加速）
def ensemble_predict_bos_bom(
    models: List[torch.nn.Module],
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    bos_list = []
    bom_list = []

    for m in models:
        m.eval()  # 确保模型在eval模式（关闭Dropout/BN训练模式）
        logits_bos, logits_bom = m(x1, x2)
        bos_list.append(F.softmax(logits_bos, dim=-1))
        bom_list.append(F.softmax(logits_bom, dim=-1))

    bos_arr = torch.stack(bos_list, dim=0)
    bom_arr = torch.stack(bom_list, dim=0)

    return {
        "bos_mean": bos_arr.mean(0),  # 集成均值（最终预测概率）
        "bos_std":  bos_arr.std(0),   # 集成标准差（不确定性）
        "bom_mean": bom_arr.mean(0),
        "bom_std":  bom_arr.std(0),
    }

def infer_singlepad_batch_ensemble(
    csv_path: str,
    root_path: str,
    models: List[MPB3net],
    part_name: str = "singlepad",
    batch_size: int = 32,
) -> pd.DataFrame:
    cfg = PART_CONFIG[part_name]
    df = pd.read_csv(csv_path)

    # 过滤指定part_name的样本
    if "part_name" in df.columns:
        df = df[df["part_name"] == part_name].reset_index(drop=True)

    results = []
    num_samples = len(df)
    print(f"[*] 一共 {num_samples} 条样本，开始 Deep Ensemble 推理 (batch_size={batch_size})")

    with tqdm(total=num_samples, desc=f"{part_name} DeepEnsemble", unit="img") as pbar:
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_df = df.iloc[start:end]

            img1_list, img2_list = [], []
            meta_list = []

            for _, row in batch_df.iterrows():
                sample_id = row["id"]
                ref_rel = row["ref_image"]
                insp_rel = row["insp_image"]

                ref_path = os.path.join(root_path, ref_rel)
                insp_path = os.path.join(root_path, insp_rel)

                # 图像预处理
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

            # 批量张量拼接并移至设备
            x1 = torch.cat(img1_list, dim=0).to(DEVICE)
            x2 = torch.cat(img2_list, dim=0).to(DEVICE)

            # 集成推理
            ens_out = ensemble_predict_bos_bom(models, x1, x2)
            bom_mean = ens_out["bom_mean"]
            bom_std  = ens_out["bom_std"]

            # 转numpy用于保存
            bom_mean_np = bom_mean.cpu().numpy()
            bom_std_np  = bom_std.cpu().numpy()

            # 整理每条样本的结果
            for i, meta in enumerate(meta_list):
                probs = bom_mean_np[i]
                stds  = bom_std_np[i]

                row_result = {
                    **meta,
                    "NONE_CONF_ENSEMBLE":                    float(probs[0]),
                    "INSUFFICIENT_SOLDER_CONF_ENSEMBLE":     float(probs[1]),
                    "PSEUDO_SOLDER_CONF_ENSEMBLE":           float(probs[2]),
                    "NONE_CONF_ENSEMBLE_STD":                float(stds[0]),
                    "INSUFFICIENT_SOLDER_CONF_ENSEMBLE_STD": float(stds[1]),
                    "PSEUDO_SOLDER_CONF_ENSEMBLE_STD":       float(stds[2]),
                }
                results.append(row_result)

            pbar.update(len(batch_df))

    # 整理结果DataFrame
    result_df = pd.DataFrame(results)
    cols = [
        "id",
        "ref_path", "insp_path",
        "NONE_CONF_ENSEMBLE",
        "INSUFFICIENT_SOLDER_CONF_ENSEMBLE",
        "PSEUDO_SOLDER_CONF_ENSEMBLE",
        "NONE_CONF_ENSEMBLE_STD",
        "INSUFFICIENT_SOLDER_CONF_ENSEMBLE_STD",
        "PSEUDO_SOLDER_CONF_ENSEMBLE_STD",
    ]
    result_df = result_df[cols]
    return result_df

# ==========================
# 评估工具函数
# ==========================
def confusion_matrix_3cls(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_class: int = N_CLASSES,
):
    """计算3分类混淆矩阵"""
    cm = np.zeros((n_class, n_class), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def calc_precision_recall_f1_binary(
    y_true_bin: np.ndarray,
    scores_pos: np.ndarray,
    thresholds: np.ndarray,
):
    """计算二分类的精确率、召回率、F1值（多阈值）"""
    precisions, recalls, f1s = [], [], []

    for thr in thresholds:
        y_pred_bin = (scores_pos >= thr).astype(np.int64)

        tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
        fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
        fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

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
    """从混淆矩阵计算精确率、召回率、F1值"""
    C = cm.shape[0]
    per_prec = np.zeros(C, dtype=float)
    per_rec = np.zeros(C, dtype=float)
    per_f1 = np.zeros(C, dtype=float)
    supports = cm.sum(axis=1)
    total_support = supports.sum()

    weighted_f1_num = 0.0

    for c in range(C):
        tp = cm[c, c]
        support = supports[c]
        pred_pos = cm[:, c].sum()

        rec = tp / support if support > 0 else float("nan")
        prec = tp / pred_pos if pred_pos > 0 else float("nan")
        
        if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
            f1 = float("nan")
        else:
            f1 = 2 * prec * rec / (prec + rec)

        per_prec[c], per_rec[c], per_f1[c] = prec, rec, f1

        if not np.isnan(f1):
            weighted_f1_num += f1 * support

    weighted_f1 = weighted_f1_num / total_support if total_support > 0 else float("nan")

    return per_prec, per_rec, per_f1, supports, weighted_f1

def compute_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = N_BINS_ECE,
):
    """计算预期校准误差（ECE）"""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)
    conf = np.clip(conf, 0.0, 1.0)
    assert conf.shape == corr.shape

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = np.zeros(n_bins, dtype=float)
    bin_acc = np.zeros(n_bins, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    bin_ids = np.digitize(conf, bin_edges[1:-1], right=True)

    for b in range(n_bins):
        mask = (bin_ids == b)
        cnt = int(mask.sum())
        bin_counts[b] = cnt
        if cnt > 0:
            bin_conf[b] = float(conf[mask].mean())
            bin_acc[b] = float(corr[mask].mean())

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
    """绘制可靠性图（校准曲线）"""
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

    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] reliability diagram saved to: {out_path}")

def compute_multiclass_vector_ece(
    softmaxes: np.ndarray,
    labels: np.ndarray,
    n_bins: int = N_BINS_ECE,
):
    """计算多分类向量ECE"""
    probs = np.asarray(softmaxes, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    N, C = probs.shape

    pred = probs.argmax(axis=1)
    max_conf = probs[np.arange(N), pred]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    acc_means = np.full((C, n_bins), np.nan, dtype=np.float64)
    conf_means = np.full((C, n_bins), np.nan, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=int)
    gap_per_bin = np.zeros(n_bins, dtype=np.float64)

    ece = 0.0
    mce = 0.0

    for b, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (max_conf > lower) & (max_conf <= upper)
        count_b = int(in_bin.sum())
        bin_counts[b] = count_b
        if count_b == 0:
            continue

        g_b = 0.0
        for k in range(C):
            in_bin_k = in_bin & (pred == k)
            count_bk = int(in_bin_k.sum())
            if count_bk == 0:
                continue

            acc_k = (labels[in_bin_k] == k).mean()
            conf_k = probs[in_bin_k, k].mean()

            acc_means[k, b] = acc_k
            conf_means[k, b] = conf_k

            gap = abs(acc_k - conf_k)
            weight_in_bin = count_bk / count_b
            g_b += weight_in_bin * gap
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
    """绘制多分类向量gap图"""
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
    """绘制单类别的多分类向量gap图"""
    acc = acc_means[class_idx]
    conf = conf_means[class_idx]
    gap = np.abs(acc - conf)

    n_bins = len(gap)
    width = bin_edges[1] - bin_edges[0]
    centers = bin_edges[:-1] + width / 2.0

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
    plt.ylim(0.0, 1.0)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] multiclass vector per-class gap (class={class_name}) saved to: {out_path}")

def eval_singlepad_deepensemble(
    csv_path: str,
    save_dir: str,
    part_name: str = "singlepad",
):
    """评估Deep Ensemble的推理结果"""
    df = pd.read_csv(csv_path)

    # 过滤指定part_name的样本
    if "part_name" in df.columns:
        df = df[df["part_name"] == part_name].reset_index(drop=True)

    if len(df) == 0:
        print(f"[WARN] CSV 中没有 part_name={part_name} 的数据，跳过评估")
        return

    # 检查GT列是否存在
    gt_cols = ["NONE_CONF", "INSUFFICIENT_SOLDER_CONF", "PSEUDO_SOLDER_CONF"]
    if not all(col in df.columns for col in gt_cols):
        raise ValueError(
            "CSV 缺少 ground truth 列："
            "NONE_CONF / INSUFFICIENT_SOLDER_CONF / PSEUDO_SOLDER_CONF"
        )

    # 提取真实标签
    soft_gt = df[gt_cols].to_numpy(dtype=np.float32)
    y_true_3 = soft_gt.argmax(axis=1).astype(np.int64)
    y_true_bin = (y_true_3 != 0).astype(np.int64)  # 二分类：ok(0) vs 缺陷(1)

    # 定义评估方法
    methods = {
        "DeepEnsemble": {
            "cols": [
                "NONE_CONF_ENSEMBLE",
                "INSUFFICIENT_SOLDER_CONF_ENSEMBLE",
                "PSEUDO_SOLDER_CONF_ENSEMBLE",
            ],
            "color": "C0",
            "marker": "o",
        },
    }

    thresholds_curve = np.linspace(0.0, 1.0, N_THRESH)
    f1_curves = {}

    print("\n===== singlepad Deep Ensemble evaluation (3-class & binary + calibration) =====")

    for method_name, cfg in methods.items():
        prob_cols = cfg["cols"]
        if not all(col in df.columns for col in prob_cols):
            print(f"[WARN] 跳过 {method_name}，缺少列: {prob_cols}")
            continue

        # 提取预测概率
        probs = df[prob_cols].to_numpy(dtype=np.float32)
        y_pred_3 = probs.argmax(axis=1)
        acc_mclass = (y_pred_3 == y_true_3).mean()

        # 多分类指标
        cm_multi = confusion_matrix_3cls(y_true_3, y_pred_3, n_class=3)
        per_prec_multi, per_rec_multi, per_f1_multi, support_multi, weighted_f1_multi = metrics_from_cm(cm_multi)

        # 二分类指标（缺陷vs正常）
        p_ok = probs[:, 0]
        p_defect = 1.0 - p_ok
        scores_defect = p_defect

        _, _, f1s_curve, _ = calc_precision_recall_f1_binary(y_true_bin, scores_defect, thresholds_curve)
        f1_curves[method_name] = {
            "f1": f1s_curve,
            "color": cfg["color"],
            "marker": cfg["marker"],
        }

        # 不同阈值下的评估
        for thr in THRESH_LIST:
            print(f"\nThreshold = {thr:.2f}  --  [{method_name}]")
            print(f"---- threshold = {thr:.2f} ----")

            y_pred_bin = (scores_defect >= thr).astype(np.int64)
            acc_bin = (y_pred_bin == y_true_bin).mean()

            cm_bin = confusion_matrix_3cls(y_true_bin, y_pred_bin, n_class=2)
            per_prec_bin, per_rec_bin, per_f1_bin, support_bin, weighted_f1_bin = metrics_from_cm(cm_bin)

            print(f"binary accuracy = {acc_bin:.5f}")
            print(f"mclass  accuracy = {acc_mclass:.5f}")

            # 二分类结果
            print("== Binary output ==")
            print(f"weighted f1 score = {weighted_f1_bin}")
            bin_names = ["ok", "defect"]
            for c in range(2):
                name = bin_names[c]
                n_c = int(support_bin[c])
                prec_c = per_prec_bin[c]
                rec_c = per_rec_bin[c]
                f1_c = per_f1_bin[c]
                omission = 1.0 - rec_c if not np.isnan(rec_c) else 0.0

                print(
                    f"{name} (n={n_c}): "
                    f"precision:{0.0 if np.isnan(prec_c) else prec_c:.3f}, "
                    f"recall:{0.0 if np.isnan(rec_c) else rec_c:.3f}, "
                    f"f1_score:{0.0 if np.isnan(f1_c) else f1_c:.3f}, "
                    f"omission_rate:{omission:.3f}"
                )

            # 多分类结果
            print("== Multiclass output ==")
            print(f"weighted f1 score = {weighted_f1_multi}")
            for c in range(N_CLASSES):
                name = CLASS_ORDER[c]
                n_c = int(support_multi[c])
                prec_c = per_prec_multi[c]
                rec_c = per_rec_multi[c]
                f1_c = per_f1_multi[c]
                omission = 1.0 - rec_c if not np.isnan(rec_c) else 0.0

                print(
                    f"{name} (n={n_c}): "
                    f"precision:{0.0 if np.isnan(prec_c) else prec_c:.3f}, "
                    f"recall:{0.0 if np.isnan(rec_c) else rec_c:.3f}, "
                    f"f1_score:{0.0 if np.isnan(f1_c) else f1_c:.3f}, "
                    f"omission_rate:{omission:.3f}"
                )

        # --------------------------
        # 二分类校准评估
        # --------------------------
        y_pred_bin_ece = (p_defect >= 0.5).astype(np.int64)
        conf_bin_pred = np.where(y_pred_bin_ece == 1, p_defect, p_ok)
        correct_bin = (y_pred_bin_ece == y_true_bin).astype(np.int64)

        # 整体二分类ECE
        ece_bin_overall, bin_edges_b, _, bin_acc_b, _ = compute_ece(conf_bin_pred, correct_bin, n_bins=N_BINS_ECE)
        bin_overall_path = os.path.join(save_dir, f"singlepad_binary_reliability_overall_{method_name}.png")
        plot_reliability_bars(bin_edges_b, bin_acc_b, f"Binary reliability overall (top1, {method_name})", bin_overall_path)

        # OK类ECE
        conf_ok = p_ok
        corr_ok = (y_true_bin == 0).astype(np.int64)
        ece_ok, bin_edges_ok, _, bin_acc_ok, _ = compute_ece(conf_ok, corr_ok, n_bins=N_BINS_ECE)
        path_ok = os.path.join(save_dir, f"singlepad_binary_reliability_true_ok_{method_name}.png")
        plot_reliability_bars(bin_edges_ok, bin_acc_ok, f"Binary reliability (true=ok, {method_name})", path_ok)

        # 缺陷类ECE
        conf_def = p_defect
        corr_def = (y_true_bin == 1).astype(np.int64)
        ece_def, bin_edges_def, _, bin_acc_def, _ = compute_ece(conf_def, corr_def, n_bins=N_BINS_ECE)
        path_def = os.path.join(save_dir, f"singlepad_binary_reliability_true_defect_{method_name}.png")
        plot_reliability_bars(bin_edges_def, bin_acc_def, f"Binary reliability (true=defect, {method_name})", path_def)

        print(f"\n[{method_name}] Binary calibration")
        print(f"  overall ECE (top1) = {ece_bin_overall:.4f}")
        print(f"  classwise ECE (prob-wise, by true class): ok={ece_ok:.4f}, defect={ece_def:.4f}")

        # --------------------------
        # 多分类校准评估（Top1）
        # --------------------------
        conf_multi_pred = probs[np.arange(len(probs)), y_pred_3]
        correct_multi = (y_pred_3 == y_true_3).astype(np.int64)

        # 整体多分类ECE
        ece_multi_overall, bin_edges_m, _, bin_acc_m, _ = compute_ece(conf_multi_pred, correct_multi, n_bins=N_BINS_ECE)
        multi_overall_path = os.path.join(save_dir, f"singlepad_multiclass_reliability_overall_{method_name}.png")
        plot_reliability_bars(bin_edges_m, bin_acc_m, f"Multiclass reliability overall (top1, {method_name})", multi_overall_path)

        # 单类别多分类ECE
        ece_multi_class = {}
        for c, cname in enumerate(CLASS_ORDER):
            conf_c = probs[:, c]
            corr_c = (y_true_3 == c).astype(np.int64)
            ece_c, bin_edges_c, _, bin_acc_c, _ = compute_ece(conf_c, corr_c, n_bins=N_BINS_ECE)
            ece_multi_class[c] = ece_c

            path_c = os.path.join(save_dir, f"singlepad_multiclass_reliability_true_{cname}_{method_name}.png")
            plot_reliability_bars(bin_edges_c, bin_acc_c, f"Multiclass reliability (true={cname}, {method_name})", path_c)

        print(f"\n[{method_name}] Multiclass calibration (true-class prob-wise)")
        print(f"  overall ECE (top1) = {ece_multi_overall:.4f}")
        print(
            f"  classwise ECE (prob-wise, by true class): "
            f"ok={ece_multi_class.get(0, float('nan')):.4f}, "
            f"undersolder={ece_multi_class.get(1, float('nan')):.4f}, "
            f"pseudosolder={ece_multi_class.get(2, float('nan')):.4f}"
        )

        # --------------------------
        # 多分类向量校准评估
        # --------------------------
        ece_vec, mce_vec, bin_edges_vec, bin_counts_vec, gap_per_bin, acc_means_vec, conf_means_vec = compute_multiclass_vector_ece(probs, y_true_3, n_bins=N_BINS_ECE)
        
        print(f"\n[{method_name}] Multiclass vector calibration (all-class probs)")
        print(f"  vector-ECE = {ece_vec:.4f}")
        print(f"  vector-MCE = {mce_vec:.4f}")

        # 绘制向量gap图
        gap_path = os.path.join(save_dir, f"singlepad_multiclass_vector_gap_{method_name}.png")
        plot_multiclass_vector_gap(bin_edges_vec, gap_per_bin, f"Multiclass vector gap per bin ({method_name})", gap_path)

        # 绘制单类别向量gap图
        for c, cname in enumerate(CLASS_ORDER):
            out_c = os.path.join(save_dir, f"singlepad_multiclass_vector_reliability_{cname}_{method_name}.png")
            plot_multiclass_vector_classwise(bin_edges_vec, acc_means_vec, conf_means_vec, c, cname, "Multiclass vector reliability", out_c)

    # --------------------------
    # 绘制F1曲线
    # --------------------------
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
    plt.title("singlepad F1 curve vs threshold (Deep Ensemble)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best")

    out_fig = os.path.join(save_dir, "singlepad_f1_curve_defect_vs_ok_ensemble.png")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nF1 曲线图已保存到: {out_fig}")

# ==========================
# 主函数
# ==========================
if __name__ == "__main__":
    part_name = "singlepad"

    # 构建Deep Ensemble模型集合
    models = _build_ensemble_for_part(
        part_name=part_name,
        max_models=8
        ,  # 可调整使用的模型数量
    )

    # 批量推理
    result_df = infer_singlepad_batch_ensemble(
        csv_path=CSV_PATH,
        root_path=ROOT_DIR,
        models=models,
        part_name=part_name,
        batch_size=64,  # 可根据显存调整
    )
    print(f"result df length: {len(result_df)}")

    # 合并推理结果到原始CSV
    orig_df = pd.read_csv(CSV_PATH)
    print(f"origin df length: {len(orig_df)}")

    # 检查ID重复
    dup_id = orig_df["id"].duplicated().sum()
    if dup_id > 0:
        raise ValueError(f"id 列出现重复: {dup_id} 行，请先清洗再推理")

    # 按ID合并结果
    orig_df = orig_df.set_index("id")
    result_df = result_df.set_index("id")
    cols = [
        "NONE_CONF_ENSEMBLE",
        "INSUFFICIENT_SOLDER_CONF_ENSEMBLE",
        "PSEUDO_SOLDER_CONF_ENSEMBLE",
        "NONE_CONF_ENSEMBLE_STD",
        "INSUFFICIENT_SOLDER_CONF_ENSEMBLE_STD",
        "PSEUDO_SOLDER_CONF_ENSEMBLE_STD",
    ]
    orig_df[cols] = result_df[cols]

    # 保存合并后的CSV
    merged = orig_df.reset_index()
    merged.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print("Deep Ensemble 结果已按 id 回写到:", CSV_PATH)

    # 评估推理结果
    eval_singlepad_deepensemble(
        csv_path=CSV_PATH,
        save_dir=SAVE_DIR,
        part_name=part_name,
    )