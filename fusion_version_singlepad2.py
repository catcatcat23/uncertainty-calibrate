#!/usr/bin/env python3
"""
贝叶斯优化搜索最优权重配比 + 完整测评 + 单个加权模型验证 +（新增）加权后 BN 校准

流程:
1. 使用贝叶斯优化搜索最优权重配比（目标：最大化 binary accuracy）
2. 保存加权平均模型（参数 soup；BN running stats 不做加权，后续校准）
3. 使用最优权重进行完整测评（保持原有测评逻辑不变）
4. 加载保存的加权模型，先做 BN 校准，再单独推理+测评（流程/输出和集成推理一致）

依赖安装:
  pip install scikit-optimize
  或
  conda install -c conda-forge scikit-optimize
"""

import os
import glob
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# ===== 新增：BN 类型 =====
from torch.nn.modules.batchnorm import _BatchNorm

# 贝叶斯优化
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from utils.utilities import TransformImage
from models.MPB3 import MPB3net

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================
# 配置
# ==========================

CKPT_DIR = "/home/cat/workspace/vlm/scripts/models/checkpoints/p2"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_CLASS = 3
N_CLASSES = 3

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
        "n_class": 3,
        "img_h": 64,
        "img_w": 64,
    },
}

# 贝叶斯优化配置
N_CALLS = 100  # 贝叶斯优化迭代次数（越大越精确但越慢）
N_INITIAL_POINTS = 10  # 初始随机采样点数
MAX_MODELS = 8
BATCH_SIZE = 64

# ===== 新增：BN 校准配置（加权模型 soup 用）=====
BN_CALIB_MAX_BATCHES = 200   # 用多少个 batch 重新统计 BN（100~500 常用）
BN_CALIB_SHUFFLE = True      # 是否打乱样本顺序做 BN 统计
BN_CALIB_USE_AMP = True      # 用 autocast 加速（只影响 forward）

# 测评配置（保持原有逻辑）
N_THRESH = 101
THRESH_LIST = [0.5, 0.8]
N_BINS_ECE = 15

SAVE_DIR = os.path.join(ROOT_DIR, "bayesian_weighted_ensemble")
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"

# ==========================
# 模型加载（健壮版本）
# ==========================

def load_mpb3_ckpt_auto(ckpt_path, backbone_arch, n_class, n_units, output_type, device):
    """模型加载：自动适配 SWA / DP / DDP / AveragedModel 等多种 ckpt 结构"""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    def _looks_like_state_dict(d: dict) -> bool:
        if not isinstance(d, dict) or len(d) == 0:
            return False
        return any(isinstance(v, torch.Tensor) for v in d.values())

    def _extract_state_dict(obj):
        if isinstance(obj, dict):
            for k in ["swa_state_dict", "avg_state_dict", "averaged_state_dict",
                      "model_state_dict", "state_dict", "model", "net", "weights"]:
                if k in obj:
                    v = obj[k]
                    if isinstance(v, torch.nn.Module):
                        return v.state_dict()
                    if _looks_like_state_dict(v):
                        return v
            if _looks_like_state_dict(obj):
                return obj
        if _looks_like_state_dict(obj):
            return obj
        raise ValueError(f"Unrecognized checkpoint format: {type(obj)}")

    def _strip_prefix(sd: dict, prefix: str) -> dict:
        out = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
            else:
                out[k] = v
        return out

    raw_sd = _extract_state_dict(ckpt)

    if isinstance(raw_sd, dict) and "n_averaged" in raw_sd:
        raw_sd = {k: v for k, v in raw_sd.items() if k != "n_averaged"}

    base_model = MPB3net(
        backbone=backbone_arch,
        pretrained=False,
        n_class=n_class,
        n_units=n_units,
        output_form=output_type,
    )

    tried = []

    def _try_load(sd: dict, tag: str):
        try:
            base_model.load_state_dict(sd, strict=True)
            print(f"=> loaded ckpt ({tag}) with strict=True: {os.path.basename(ckpt_path)}")
            return True
        except Exception as e:
            tried.append((tag, str(e)))
            return False

    if _try_load(raw_sd, "raw"):
        return base_model.to(device).eval()

    sd = raw_sd
    for i in range(3):
        if any(k.startswith("module.") for k in sd.keys()):
            sd = _strip_prefix(sd, "module.")
            if _try_load(sd, f"strip_module_x{i+1}"):
                return base_model.to(device).eval()
        else:
            break

    for prefix in ["model.", "net."]:
        if any(k.startswith(prefix) for k in raw_sd.keys()):
            sd2 = _strip_prefix(raw_sd, prefix)
            if _try_load(sd2, f"strip_{prefix[:-1]}"):
                return base_model.to(device).eval()

    try:
        incompatible = base_model.load_state_dict(sd, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        print(f"[WARN] loaded with strict=False: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("[WARN] missing_keys (first 20):", missing[:20])
        if len(unexpected) > 0:
            print("[WARN] unexpected_keys (first 20):", unexpected[:20])
    except Exception as e:
        print("[ERROR] all load attempts failed. Tried:")
        for tag, err in tried[:10]:
            print(f"  - {tag}: {err.splitlines()[-1] if err else err}")
        raise e

    return base_model.to(device).eval()


def find_checkpoints(pattern: str, max_models: int = None) -> List[str]:
    """查找 checkpoint 文件"""
    full_pattern = os.path.join(CKPT_DIR, pattern)
    files = sorted(glob.glob(full_pattern))

    if not files:
        raise FileNotFoundError(f"未找到匹配的 checkpoint: {full_pattern}")

    if max_models:
        files = files[-max_models:]

    print(f"找到 {len(files)} 个 checkpoint:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    return files


def load_models(ckpt_paths: List[str], cfg: Dict) -> List[MPB3net]:
    """加载多个模型（使用健壮的 auto 加载函数）"""
    models = []

    for ckpt_path in ckpt_paths:
        print(f"\n=> 加载: {os.path.basename(ckpt_path)}")
        model = load_mpb3_ckpt_auto(
            ckpt_path=ckpt_path,
            backbone_arch=cfg["backbone"],
            n_class=cfg["n_class"],
            n_units=cfg["n_units"],
            output_type="dual2",
            device=DEVICE,
        )
        models.append(model)

    print(f"\n✓ 成功加载 {len(models)} 个模型")
    return models


# ==========================
# 加权推理 + 单个模型推理
# ==========================

@torch.no_grad()
def weighted_predict(
    models: List[torch.nn.Module],
    weights: np.ndarray,
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> torch.Tensor:
    """加权 ensemble 预测（对 softmax 概率做加权）"""
    weights = weights / weights.sum()
    weights_tensor = torch.FloatTensor(weights).to(DEVICE)

    bom_weighted = None

    for i, model in enumerate(models):
        _, logits_bom = model(x1, x2)
        prob_bom = F.softmax(logits_bom, dim=-1)

        if bom_weighted is None:
            bom_weighted = weights_tensor[i] * prob_bom
        else:
            bom_weighted += weights_tensor[i] * prob_bom

    return bom_weighted


@torch.no_grad()
def single_model_predict(
    model: torch.nn.Module,
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> torch.Tensor:
    """单个模型预测（复用原有前向逻辑）"""
    _, logits_bom = model(x1, x2)
    prob_bom = F.softmax(logits_bom, dim=-1)
    return prob_bom


def infer_with_weights(
    models: List[MPB3net],
    weights: np.ndarray,
    csv_path: str,
    root_path: str,
    cfg: Dict,
    part_name: str,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用给定权重做集成推理，返回 (probs, y_true)"""
    df = pd.read_csv(csv_path)

    if "part_name" in df.columns:
        df = df[df["part_name"] == part_name].reset_index(drop=True)

    gt_cols = ["NONE_CONF", "INSUFFICIENT_SOLDER_CONF", "PSEUDO_SOLDER_CONF"]
    soft_gt = df[gt_cols].to_numpy(dtype=np.float32)
    y_true = soft_gt.argmax(axis=1)

    all_probs = []

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end]

        img1_list, img2_list = [], []

        for _, row in batch_df.iterrows():
            ref_path = os.path.join(root_path, row["ref_image"])
            insp_path = os.path.join(root_path, row["insp_image"])

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

            img1_list.append(torch.FloatTensor(ref_img))
            img2_list.append(torch.FloatTensor(insp_img))

        x1 = torch.cat(img1_list, dim=0).to(DEVICE)
        x2 = torch.cat(img2_list, dim=0).to(DEVICE)

        probs = weighted_predict(models, weights, x1, x2)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    return all_probs, y_true


def infer_single_weighted_model(
    model: MPB3net,
    csv_path: str,
    root_path: str,
    cfg: Dict,
    part_name: str,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """单个加权模型推理（复用所有数据处理逻辑，仅改预测函数）"""
    df = pd.read_csv(csv_path)

    if "part_name" in df.columns:
        df = df[df["part_name"] == part_name].reset_index(drop=True)

    gt_cols = ["NONE_CONF", "INSUFFICIENT_SOLDER_CONF", "PSEUDO_SOLDER_CONF"]
    soft_gt = df[gt_cols].to_numpy(dtype=np.float32)
    y_true = soft_gt.argmax(axis=1)

    all_probs = []

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end]

        img1_list, img2_list = [], []

        for _, row in batch_df.iterrows():
            ref_path = os.path.join(root_path, row["ref_image"])
            insp_path = os.path.join(root_path, row["insp_image"])

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

            img1_list.append(torch.FloatTensor(ref_img))
            img2_list.append(torch.FloatTensor(insp_img))

        x1 = torch.cat(img1_list, dim=0).to(DEVICE)
        x2 = torch.cat(img2_list, dim=0).to(DEVICE)

        probs = single_model_predict(model, x1, x2)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    return all_probs, y_true


# ==========================
# 新增：加权模型 BN 校准
# ==========================

def _set_bn_train_others_eval(model: torch.nn.Module):
    """只让 BN 处于 train（更新 running stats），其余层保持 eval（避免 dropout 干扰）"""
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.train()
        else:
            m.eval()


@torch.no_grad()
def bn_calibrate_from_csv(
    model: MPB3net,
    csv_path: str,
    root_path: str,
    cfg: Dict,
    part_name: str,
    batch_size: int = 64,
    max_batches: int = 200,
    shuffle: bool = True,
    use_amp: bool = True,
):
    """
    用 CSV 数据跑 forward，重新统计 BN running_mean/running_var
    - 不反传
    - 只更新 BN buffer
    - 双输入 model(x1, x2)
    """
    print(f"[BN-CALIB] start: max_batches={max_batches}, batch_size={batch_size}, shuffle={shuffle}")

    df = pd.read_csv(csv_path)
    if "part_name" in df.columns:
        df = df[df["part_name"] == part_name].reset_index(drop=True)

    if shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 1) BN train, others eval
    _set_bn_train_others_eval(model)

    # 2) reset BN stats + save original momentum
    momenta = {}
    for m in model.modules():
        if isinstance(m, _BatchNorm) and getattr(m, "track_running_stats", True):
            momenta[m] = m.momentum
            if hasattr(m, "running_mean") and m.running_mean is not None:
                m.running_mean.zero_()
            if hasattr(m, "running_var") and m.running_var is not None:
                m.running_var.fill_(1)
            if hasattr(m, "num_batches_tracked") and m.num_batches_tracked is not None:
                m.num_batches_tracked.zero_()
            m.momentum = 0.0  # 后面用动态 momentum

    n_seen = 0
    batch_count = 0

    for bi in range(0, len(df), batch_size):
        if max_batches is not None and batch_count >= max_batches:
            break

        batch_df = df.iloc[bi:bi + batch_size]
        img1_list, img2_list = [], []

        for _, row in batch_df.iterrows():
            ref_path = os.path.join(root_path, row["ref_image"])
            insp_path = os.path.join(root_path, row["insp_image"])

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

            img1_list.append(torch.FloatTensor(ref_img))
            img2_list.append(torch.FloatTensor(insp_img))

        x1 = torch.cat(img1_list, dim=0).to(DEVICE, non_blocking=True)
        x2 = torch.cat(img2_list, dim=0).to(DEVICE, non_blocking=True)

        bsz = int(x1.shape[0])
        momentum = bsz / float(n_seen + bsz) if (n_seen + bsz) > 0 else 1.0
        for m in momenta.keys():
            m.momentum = momentum

        # autocast 加速（只做 forward）
        if DEVICE.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(use_amp and DEVICE.type == "cuda"),
            ):
                model(x1, x2)
        else:
            model(x1, x2)

        n_seen += bsz
        batch_count += 1

    # 3) restore momentum + set eval
    for m, mom in momenta.items():
        m.momentum = mom
    model.eval()

    print(f"[BN-CALIB] done: seen={n_seen} samples, batches={batch_count}")


# ==========================
# 贝叶斯优化
# ==========================

def bayesian_search(
    models: List[MPB3net],
    csv_path: str,
    root_path: str,
    cfg: Dict,
    part_name: str,
    n_calls: int,
    n_initial_points: int,
    batch_size: int,
) -> Tuple[np.ndarray, float, List[Dict]]:
    """使用贝叶斯优化搜索最优权重"""
    n_models = len(models)

    print(f"\n{'='*60}")
    print(f"开始贝叶斯优化搜索")
    print(f"{'='*60}")
    print(f"模型数量: {n_models}")
    print(f"优化迭代次数: {n_calls}")
    print(f"初始随机点数: {n_initial_points}")
    print(f"目标: 最大化 binary accuracy @ threshold=0.5")
    print(f"{'='*60}\n")

    space = [Real(0.0, 1.0, name=f'w{i}') for i in range(n_models)]

    all_results = []
    best_acc = 0.0
    best_weights = None

    @use_named_args(space)
    def objective(**kwargs):
        nonlocal best_acc, best_weights, all_results

        weights = np.array([kwargs[f'w{i}'] for i in range(n_models)])

        if weights.sum() == 0:
            return 1.0

        weights = weights / weights.sum()

        probs, y_true = infer_with_weights(
            models, weights, csv_path, root_path, cfg, part_name, batch_size
        )

        y_true_bin = (y_true != 0).astype(np.int64)
        p_defect = 1.0 - probs[:, 0]
        y_pred_bin = (p_defect >= 0.5).astype(np.int64)
        acc = (y_pred_bin == y_true_bin).mean()

        result = {
            "weights": weights.tolist(),
            "binary_accuracy": float(acc),
        }
        all_results.append(result)

        if acc > best_acc:
            best_acc = acc
            best_weights = weights.copy()
            print(f"✓ 新最优解! binary_acc={acc:.6f}, weights={weights}")

        return -acc

    print("开始优化...\n")
    _ = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=42,
        verbose=False,
    )

    print(f"\n{'='*60}")
    print(f"优化完成!")
    print(f"{'='*60}")
    print(f"最优 binary accuracy: {best_acc:.6f}")
    print(f"最优权重: {best_weights}")
    print(f"{'='*60}\n")

    return best_weights, best_acc, all_results


# ==========================
# 保存模型（修改：BN running stats 不做加权）
# ==========================

def save_weighted_model(
    models: List[MPB3net],
    weights: np.ndarray,
    save_path: str,
    metadata: Dict = None,
):
    """保存加权平均后的模型（参数 soup）"""
    weights = weights / weights.sum()

    weighted_state_dict = {}
    all_state_dicts = [model.state_dict() for model in models]
    base_state = all_state_dicts[0]

    for name in base_state.keys():
        base_param = base_state[name]

        # ===== 关键：BN 的 running stats 不做加权，直接用第一个模型的 =====
        if name.endswith("running_mean") or name.endswith("running_var"):
            weighted_state_dict[name] = base_param.clone()
            continue

        # 跳过非浮点数参数（如 num_batches_tracked）
        if not base_param.dtype.is_floating_point:
            weighted_state_dict[name] = base_param.clone()
            continue

        weighted_param = torch.zeros_like(base_param)

        for i, state_dict in enumerate(all_state_dicts):
            if name in state_dict:
                weighted_param += weights[i] * state_dict[name]
            else:
                print(f"[WARN] 模型 {i} 缺少参数: {name}")

        weighted_state_dict[name] = weighted_param

    save_dict = {
        "state_dict": weighted_state_dict,
        "ensemble_weights": weights.tolist(),
        "n_models": len(models),
    }

    if metadata:
        save_dict.update(metadata)

    torch.save(save_dict, save_path)
    print(f"✓ 加权模型已保存: {os.path.basename(save_path)}")


# ==========================
# 原有测评逻辑（完全保持不变）
# ==========================

def confusion_matrix_3cls(y_true: np.ndarray, y_pred: np.ndarray, n_class: int = N_CLASSES):
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

        prec = tp / (tp + fp) if tp + fp > 0 else 1.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

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

        if np.isnan(prec) or np.isnan(rec) or prec + rec == 0:
            f1 = float("nan")
        else:
            f1 = 2 * prec * rec / (prec + rec)

        per_prec[c], per_rec[c], per_f1[c] = prec, rec, f1

        if not np.isnan(f1):
            weighted_f1_num += f1 * support

    weighted_f1 = weighted_f1_num / total_support if total_support > 0 else float("nan")

    return per_prec, per_rec, per_f1, supports, weighted_f1


def compute_ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = N_BINS_ECE):
    conf = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
    corr = np.asarray(correctness, dtype=np.float64)

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
    ece = sum((bin_counts[b] / N) * abs(bin_acc[b] - bin_conf[b])
              for b in range(n_bins) if bin_counts[b] > 0)

    return ece, bin_edges, bin_conf, bin_acc, bin_counts


def plot_reliability_bars(bin_edges: np.ndarray, bin_acc: np.ndarray, title: str, out_path: str):
    width = bin_edges[1] - bin_edges[0]
    centers = bin_edges[:-1] + width / 2.0

    plt.figure(figsize=(5, 5))
    plt.bar(centers, bin_acc, width=width * 0.9, align="center", alpha=0.7,
            edgecolor="black", label="Empirical accuracy")
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


def compute_multiclass_vector_ece(softmaxes: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS_ECE):
    probs = np.asarray(softmaxes, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    N, C = probs.shape

    pred = probs.argmax(axis=1)
    max_conf = probs[np.arange(N), pred]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers, bin_uppers = bin_edges[:-1], bin_edges[1:]

    acc_means = np.full((C, n_bins), np.nan, dtype=np.float64)
    conf_means = np.full((C, n_bins), np.nan, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=int)
    gap_per_bin = np.zeros(n_bins, dtype=np.float64)

    ece, mce = 0.0, 0.0

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
            g_b += (count_bk / count_b) * gap
            ece += (count_bk / N) * gap

        gap_per_bin[b] = g_b
        mce = max(mce, g_b)

    return ece, mce, bin_edges, bin_counts, gap_per_bin, acc_means, conf_means


def plot_multiclass_vector_gap(bin_edges: np.ndarray, gap_per_bin: np.ndarray, title: str, out_path: str):
    width = bin_edges[1] - bin_edges[0]
    centers = bin_edges[:-1] + width / 2.0

    plt.figure(figsize=(6, 4))
    plt.bar(centers, gap_per_bin, width=width * 0.9, align="center", alpha=0.7,
            edgecolor="black", label=r"$\sum_k |acc_k - conf_k|$")
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
    bin_edges: np.ndarray, acc_means: np.ndarray, conf_means: np.ndarray,
    class_idx: int, class_name: str, title_prefix: str, out_path: str
):
    acc = acc_means[class_idx]
    conf = conf_means[class_idx]
    gap = np.abs(acc - conf)

    width = bin_edges[1] - bin_edges[0]
    centers = bin_edges[:-1] + width / 2.0
    mask = ~np.isnan(gap)

    plt.figure(figsize=(5, 5))
    plt.bar(centers[mask], gap[mask], width=width * 0.9, align="center", alpha=0.7,
            edgecolor="black", label=r"|acc - conf| (class=%s)" % class_name)
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


def eval_weighted_ensemble(
    probs: np.ndarray,
    y_true_3: np.ndarray,
    save_dir: str,
    method_name: str = "BayesianWeightedEnsemble"
):
    """使用原有测评逻辑进行完整评估（完全复用，仅改method_name区分来源）"""

    print(f"\n===== singlepad {method_name} evaluation (3-class & binary + calibration) =====")

    y_true_bin = (y_true_3 != 0).astype(np.int64)
    y_pred_3 = probs.argmax(axis=1)
    acc_mclass = (y_pred_3 == y_true_3).mean()

    cm_multi = confusion_matrix_3cls(y_true_3, y_pred_3, n_class=3)
    per_prec_multi, per_rec_multi, per_f1_multi, support_multi, weighted_f1_multi = metrics_from_cm(cm_multi)

    p_ok = probs[:, 0]
    p_defect = 1.0 - p_ok

    thresholds_curve = np.linspace(0.0, 1.0, N_THRESH)
    _, _, f1s_curve, _ = calc_precision_recall_f1_binary(y_true_bin, p_defect, thresholds_curve)

    for thr in THRESH_LIST:
        print(f"\nThreshold = {thr:.2f}  --  [{method_name}]")
        print(f"---- threshold = {thr:.2f} ----")

        y_pred_bin = (p_defect >= thr).astype(np.int64)
        acc_bin = (y_pred_bin == y_true_bin).mean()

        cm_bin = confusion_matrix_3cls(y_true_bin, y_pred_bin, n_class=2)
        per_prec_bin, per_rec_bin, per_f1_bin, support_bin, weighted_f1_bin = metrics_from_cm(cm_bin)

        print(f"binary accuracy = {acc_bin:.5f}")
        print(f"mclass  accuracy = {acc_mclass:.5f}")

        print("== Binary output ==")
        print(f"weighted f1 score = {weighted_f1_bin}")

        bin_names = ["ok", "defect"]
        for c in range(2):
            name = bin_names[c]
            n_c = int(support_bin[c])
            prec_c, rec_c, f1_c = per_prec_bin[c], per_rec_bin[c], per_f1_bin[c]
            omission = 0.0 if np.isnan(rec_c) else 1.0 - rec_c
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
            prec_c, rec_c, f1_c = per_prec_multi[c], per_rec_multi[c], per_f1_multi[c]
            omission = 0.0 if np.isnan(rec_c) else 1.0 - rec_c
            print(
                f"{name} (n={n_c}): "
                f"precision:{0.0 if np.isnan(prec_c) else prec_c:.3f}, "
                f"recall:{0.0 if np.isnan(rec_c) else rec_c:.3f}, "
                f"f1_score:{0.0 if np.isnan(f1_c) else f1_c:.3f}, "
                f"omission_rate:{omission:.3f}"
            )

    # ====== Binary calibration ======
    y_pred_bin_ece = (p_defect >= 0.5).astype(np.int64)
    conf_bin_pred = np.where(y_pred_bin_ece == 1, p_defect, p_ok)
    correct_bin = (y_pred_bin_ece == y_true_bin).astype(np.int64)

    ece_bin_overall, bin_edges_b, _, bin_acc_b, _ = compute_ece(conf_bin_pred, correct_bin)
    plot_reliability_bars(
        bin_edges_b, bin_acc_b,
        f"Binary reliability overall (top1, {method_name})",
        os.path.join(save_dir, f"singlepad_binary_reliability_overall_{method_name}.png")
    )

    ece_ok, bin_edges_ok, _, bin_acc_ok, _ = compute_ece(p_ok, (y_true_bin == 0).astype(np.int64))
    plot_reliability_bars(
        bin_edges_ok, bin_acc_ok,
        f"Binary reliability (true=ok, {method_name})",
        os.path.join(save_dir, f"singlepad_binary_reliability_true_ok_{method_name}.png")
    )

    ece_def, bin_edges_def, _, bin_acc_def, _ = compute_ece(p_defect, (y_true_bin == 1).astype(np.int64))
    plot_reliability_bars(
        bin_edges_def, bin_acc_def,
        f"Binary reliability (true=defect, {method_name})",
        os.path.join(save_dir, f"singlepad_binary_reliability_true_defect_{method_name}.png")
    )

    print(f"\n[{method_name}] Binary calibration")
    print(f"  overall ECE (top1) = {ece_bin_overall:.4f}")
    print(f"  classwise ECE (prob-wise, by true class): ok={ece_ok:.4f}, defect={ece_def:.4f}")

    # ====== Multiclass calibration：top1 + prob-wise ======
    conf_multi_pred = probs[np.arange(len(probs)), y_pred_3]
    correct_multi = (y_pred_3 == y_true_3).astype(np.int64)

    ece_multi_overall, bin_edges_m, _, bin_acc_m, _ = compute_ece(conf_multi_pred, correct_multi)
    plot_reliability_bars(
        bin_edges_m, bin_acc_m,
        f"Multiclass reliability overall (top1, {method_name})",
        os.path.join(save_dir, f"singlepad_multiclass_reliability_overall_{method_name}.png")
    )

    ece_multi_class = {}
    for c, cname in enumerate(CLASS_ORDER):
        conf_c = probs[:, c]
        corr_c = (y_true_3 == c).astype(np.int64)

        ece_c, bin_edges_c, _, bin_acc_c, _ = compute_ece(conf_c, corr_c)
        ece_multi_class[c] = ece_c

        plot_reliability_bars(
            bin_edges_c, bin_acc_c,
            f"Multiclass reliability (true={cname}, {method_name})",
            os.path.join(save_dir, f"singlepad_multiclass_reliability_true_{cname}_{method_name}.png")
        )

    print(f"\n[{method_name}] Multiclass calibration (true-class prob-wise)")
    print(f"  overall ECE (top1) = {ece_multi_overall:.4f}")
    print(
        f"  classwise ECE (prob-wise, by true class): "
        f"ok={ece_multi_class.get(0, float('nan')):.4f}, "
        f"undersolder={ece_multi_class.get(1, float('nan')):.4f}, "
        f"pseudosolder={ece_multi_class.get(2, float('nan')):.4f}"
    )

    # ====== Multiclass "向量版" ECE ======
    ece_vec, mce_vec, bin_edges_vec, bin_counts_vec, gap_per_bin, acc_means_vec, conf_means_vec = \
        compute_multiclass_vector_ece(probs, y_true_3)

    print(f"\n[{method_name}] Multiclass vector calibration (all-class probs)")
    print(f"  vector-ECE = {ece_vec:.4f}")
    print(f"  vector-MCE = {mce_vec:.4f}")

    plot_multiclass_vector_gap(
        bin_edges_vec, gap_per_bin,
        f"Multiclass vector gap per bin ({method_name})",
        os.path.join(save_dir, f"singlepad_multiclass_vector_gap_{method_name}.png")
    )

    for c, cname in enumerate(CLASS_ORDER):
        plot_multiclass_vector_classwise(
            bin_edges_vec, acc_means_vec, conf_means_vec, c, cname,
            "Multiclass vector reliability",
            os.path.join(save_dir, f"singlepad_multiclass_vector_reliability_{cname}_{method_name}.png")
        )

    # ====== F1 曲线 ======
    best_idx = int(f1s_curve.argmax())
    best_thr = thresholds_curve[best_idx]
    best_f1 = f1s_curve[best_idx]

    print(f"\n[{method_name}] F1 curve (binary defect vs ok)")
    print(f"  Max F1 = {best_f1:.4f} at threshold = {best_thr:.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(
        thresholds_curve, f1s_curve,
        label=f"{method_name} (max F1={best_f1:.3f} @ {best_thr:.2f})",
        color="C1" if "Single" in method_name else "C0"
    )
    plt.xlabel("Threshold on P(defect)")
    plt.ylabel("F1 score (defect vs ok)")
    plt.title(f"singlepad F1 curve vs threshold ({method_name})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"singlepad_f1_curve_defect_vs_ok_{method_name}.png"),
        dpi=200, bbox_inches="tight"
    )
    plt.close()
    print(f"\nF1 曲线图已保存到: {os.path.join(save_dir, f'singlepad_f1_curve_defect_vs_ok_{method_name}.png')}")


# ==========================
# 主函数（新增 BN 校准调用）
# ==========================

def main():
    part_name = "singlepad"

    print("\n" + "="*80)
    print("贝叶斯优化加权 Ensemble + 单个加权模型验证（含 BN 校准）".center(80))
    print("="*80 + "\n")

    # 1. 加载配置和模型
    cfg = PART_CONFIG[part_name]
    ckpt_paths = find_checkpoints(cfg["ckpt_pattern"], MAX_MODELS)
    models = load_models(ckpt_paths, cfg)

    # 2. 贝叶斯优化搜索最优权重
    best_weights, best_acc, all_results = bayesian_search(
        models=models,
        csv_path=CSV_PATH,
        root_path=ROOT_DIR,
        cfg=cfg,
        part_name=part_name,
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL_POINTS,
        batch_size=BATCH_SIZE,
    )

    # 3. 保存搜索结果
    print(f"\n{'='*60}")
    print(f"搜索完成！")
    print(f"{'='*60}")
    print(f"最优 binary accuracy: {best_acc:.6f}")
    print(f"最优权重: {best_weights}")
    print(f"{'='*60}\n")

    results_json = os.path.join(SAVE_DIR, "bayesian_search_results.json")
    with open(results_json, "w") as f:
        json.dump({
            "best_weights": best_weights.tolist(),
            "best_binary_accuracy": float(best_acc),
            "n_calls": N_CALLS,
            "n_initial_points": N_INITIAL_POINTS,
            "n_models": len(models),
            "checkpoints": [os.path.basename(p) for p in ckpt_paths],
            "all_results": all_results,
        }, f, indent=2)
    print(f"✓ 搜索结果: {results_json}")

    weights_txt = os.path.join(SAVE_DIR, "best_weights.txt")
    with open(weights_txt, "w") as f:
        f.write(f"Best Binary Accuracy: {best_acc:.6f}\n")
        f.write(f"Best Weights: {best_weights}\n\n")
        f.write("Checkpoints:\n")
        for i, (ckpt, w) in enumerate(zip(ckpt_paths, best_weights)):
            f.write(f"  [{i}] {os.path.basename(ckpt)}: {w:.6f}\n")
    print(f"✓ 权重信息: {weights_txt}")

    # 4. 保存加权模型（参数 soup；BN running stats 不加权，后续 BN 校准）
    model_path = os.path.join(SAVE_DIR, f"bayesian_weighted_ensemble_acc{best_acc:.4f}.pth.tar")
    save_weighted_model(
        models=models,
        weights=best_weights,
        save_path=model_path,
        metadata={
            "binary_accuracy": float(best_acc),
            "part_name": part_name,
            "checkpoints": [os.path.basename(p) for p in ckpt_paths],
            "optimization_method": "bayesian",
            "note": "BN running stats are NOT averaged; calibrate BN after loading."
        }
    )
    print(f"✓ 加权模型: {model_path}")

    # 5. 使用最优权重进行集成推理和测评（原有逻辑不变）
    print(f"\n{'='*60}")
    print("使用最优权重进行集成推理完整测评...")
    print(f"{'='*60}\n")

    probs_ensemble, y_true_ensemble = infer_with_weights(
        models=models,
        weights=best_weights,
        csv_path=CSV_PATH,
        root_path=ROOT_DIR,
        cfg=cfg,
        part_name=part_name,
        batch_size=BATCH_SIZE,
    )

    eval_weighted_ensemble(probs_ensemble, y_true_ensemble, SAVE_DIR, method_name="BayesianWeightedEnsemble")

    # 6. 加载保存的加权模型 -> BN 校准 -> 单个模型推理/测评
    print(f"\n{'='*60}")
    print("加载加权模型 -> BN 校准 -> 单个模型推理完整测评...")
    print(f"{'='*60}\n")

    single_model = load_mpb3_ckpt_auto(
        ckpt_path=model_path,
        backbone_arch=cfg["backbone"],
        n_class=cfg["n_class"],
        n_units=cfg["n_units"],
        output_type="dual2",
        device=DEVICE,
    )

    # ===== 新增：BN 校准（强烈建议）=====
    bn_calibrate_from_csv(
        model=single_model,
        csv_path=CSV_PATH,
        root_path=ROOT_DIR,
        cfg=cfg,
        part_name=part_name,
        batch_size=BATCH_SIZE,
        max_batches=BN_CALIB_MAX_BATCHES,
        shuffle=BN_CALIB_SHUFFLE,
        use_amp=BN_CALIB_USE_AMP,
    )

    # 可选：保存 BN 校准后的 soup，便于复现
    bncal_model_path = model_path.replace(".pth.tar", f"_bncal{BN_CALIB_MAX_BATCHES}.pth.tar")
    torch.save(
        {
            "state_dict": single_model.state_dict(),
            "ensemble_weights": best_weights.tolist(),
            "n_models": len(models),
            "bn_calibrated": True,
            "bn_calib_max_batches": BN_CALIB_MAX_BATCHES,
        },
        bncal_model_path,
    )
    print(f"✓ BN 校准后的加权模型已保存: {bncal_model_path}")

    probs_single, y_true_single = infer_single_weighted_model(
        model=single_model,
        csv_path=CSV_PATH,
        root_path=ROOT_DIR,
        cfg=cfg,
        part_name=part_name,
        batch_size=BATCH_SIZE,
    )

    eval_weighted_ensemble(probs_single, y_true_single, SAVE_DIR, method_name="SingleWeightedModel")

    # 7. 写回 CSV（新增单个模型结果列，区分集成/单个）
    orig_df = pd.read_csv(CSV_PATH)
    print(f"\norigin df length: {len(orig_df)}")

    dup_id = orig_df["id"].duplicated().sum()
    if dup_id > 0:
        raise ValueError(f"id 列出现重复: {dup_id} 行，请先清洗再推理")

    if "part_name" in orig_df.columns:
        mask = orig_df["part_name"] == part_name
        # 集成推理结果列
        orig_df.loc[mask, "NONE_CONF_BAYESIAN_WEIGHTED"] = probs_ensemble[:, 0]
        orig_df.loc[mask, "INSUFFICIENT_SOLDER_CONF_BAYESIAN_WEIGHTED"] = probs_ensemble[:, 1]
        orig_df.loc[mask, "PSEUDO_SOLDER_CONF_BAYESIAN_WEIGHTED"] = probs_ensemble[:, 2]
        # 单个模型结果列（BN 校准后）
        orig_df.loc[mask, "NONE_CONF_SINGLE_WEIGHTED"] = probs_single[:, 0]
        orig_df.loc[mask, "INSUFFICIENT_SOLDER_CONF_SINGLE_WEIGHTED"] = probs_single[:, 1]
        orig_df.loc[mask, "PSEUDO_SOLDER_CONF_SINGLE_WEIGHTED"] = probs_single[:, 2]
    else:
        orig_df["NONE_CONF_BAYESIAN_WEIGHTED"] = probs_ensemble[:, 0]
        orig_df["INSUFFICIENT_SOLDER_CONF_BAYESIAN_WEIGHTED"] = probs_ensemble[:, 1]
        orig_df["PSEUDO_SOLDER_CONF_BAYESIAN_WEIGHTED"] = probs_ensemble[:, 2]
        orig_df["NONE_CONF_SINGLE_WEIGHTED"] = probs_single[:, 0]
        orig_df["INSUFFICIENT_SOLDER_CONF_SINGLE_WEIGHTED"] = probs_single[:, 1]
        orig_df["PSEUDO_SOLDER_CONF_SINGLE_WEIGHTED"] = probs_single[:, 2]

    orig_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✓ 集成+单个加权模型结果已写回到: {CSV_PATH}")

    # 8. 输出对比总结
    print(f"\n{'='*80}")
    print("结果总结".center(80))
    print("="*80)

    acc_ensemble_bin = (
        (((1.0 - probs_ensemble[:, 0]) >= 0.5).astype(np.int64))
        == ((y_true_ensemble != 0).astype(np.int64))
    ).mean()

    acc_single_bin = (
        (((1.0 - probs_single[:, 0]) >= 0.5).astype(np.int64))
        == ((y_true_single != 0).astype(np.int64))
    ).mean()

    print(f"\n核心指标对比 (threshold=0.5):")
    print(f"  贝叶斯集成推理 binary acc: {acc_ensemble_bin:.6f}")
    print(f"  单个加权模型推理 binary acc: {acc_single_bin:.6f}")
    print(f"  绝对差异: {abs(acc_ensemble_bin - acc_single_bin):.6f}")

    print(f"\n文件输出:")
    print(f"  加权模型(未BN校准): {model_path}")
    print(f"  加权模型(已BN校准): {bncal_model_path}")
    print(f"  搜索结果: {results_json}")
    print(f"  评估图表: {SAVE_DIR}/")
    print(f"  结果CSV: {CSV_PATH}")

    print(f"\n{'='*80}")
    print("完成！".center(80))
    print("="*80)


if __name__ == "__main__":
    main()
