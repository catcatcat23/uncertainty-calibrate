#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bayesian Weighted Ensemble vs Parameter-Soup Single Model
(对比 BN buffer 两种策略：BN=first vs BN=weighted-average；不做 BN 刷新/校准)
并且：在【两个数据集】上做【同一套 softlabel 完整测评】（和你第一份脚本的测评口径一致）

两个数据集（默认）：
1) calibrate_with_softlabels.csv  -> 只取 black + singlepad，soft_* 列
2) checked_samples.csv            -> singlepad，全量，NONE_CONF 等 soft label 列（若存在则映射成 soft_*）

输出目录：
ROOT_DIR/bayes_soup_softlabel_compare/
  ├── weights/best_weights.json
  ├── dataset_softlabel_black/...
  └── dataset_checked_samples/...

方法（每个数据集都会跑）：
A) BayesianWeightedEnsemble   : 概率层面按权重加权 (不需要 soup)
B) Soup_BNfirst              : 参数 soup + BN running stats 直接用第一个模型
C) Soup_BNwavg               : 参数 soup + BN running stats 也做权重平均

依赖：
- torch, numpy, pandas, matplotlib, tqdm
- scikit-optimize (仅当 best_weights.json 不存在且需要运行贝叶斯优化时)

注意：
- 你要求“不刷 BN”，本脚本不包含任何 bn_calibrate / update_bn 操作
- 测评逻辑：使用同一套 eval_softlabel_split()（含 ECE、vector gap、soft gap、hist、bin count 等）
"""

import os
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 你的工程依赖
from utils.utilities import TransformImage
from models.MPB3 import MPB3net

# =========================
# CONFIG
# =========================
ROOT_DIR = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310"

CSV_SOFTLABEL = os.path.join(ROOT_DIR, "calibrate_with_softlabels.csv")
CSV_CHECKED   = os.path.join(ROOT_DIR, "checked_samples.csv")

SAVE_DIR = os.path.join(ROOT_DIR, "bayes_soup_softlabel_compare")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# 分箱/测评配置
N_BINS_ECE = 15

# 模型配置（singlepad）
CFG = dict(
    backbone="fcdropoutmobilenetv3large",
    n_class=3,
    n_units=[256, 256],
    output_type="dual2",
    img_h=64,
    img_w=64,
)

# 类别/列名（统一口径，和你第一份脚本一致）
CLASS_ORDER = ["ok", "INSUFFICIENT_SOLDER", "PSEUDO_SOLDER"]
SOFT_COLS = ["soft_ok", "soft_INSUFFICIENT_SOLDER", "soft_PSEUDO_SOLDER"]
PROB_COLS = ["pred_ok", "pred_INSUFFICIENT_SOLDER", "pred_PSEUDO_SOLDER"]

OPTIONAL_SOFT_COLD = "soft_COLD_WELD"

# 你手动配置的 ckpt 列表（默认用这个；如果想改成 pattern 搜索，自行替换此列表）
ENSEMBLE_CKPT_PATHS = [
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2bestacc.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2bestbiacc.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2bestmlacc.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2last.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2swa.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2top0.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2top1.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2top2.pth.tar",
]

# 贝叶斯优化（若 weights 文件不存在会触发）
N_CALLS = 100
N_INITIAL_POINTS = 10
WEIGHTS_DIR = os.path.join(SAVE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
BEST_WEIGHTS_JSON = os.path.join(WEIGHTS_DIR, "best_weights.json")

# 字体：满足你“中文 caption 时包含 SimHei”的习惯，同时兼容没装 SimHei 的环境
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# Utils
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    s = mat.sum(axis=1, keepdims=True)
    return mat / (s + 1e-12)

def ensure_probs_like(mat: np.ndarray, name: str = "probs") -> np.ndarray:
    x = np.asarray(mat, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)

    row_sum = x.sum(axis=1)
    bad_ratio = float(np.mean(np.abs(row_sum - 1.0) > 1e-2))
    if bad_ratio > 0.2:
        print(f"[WARN] {name}: row-sum not ~1 for {bad_ratio*100:.1f}% rows, apply row-normalize.")
        x = _normalize_rows(x)

    s = x.sum(axis=1)
    bad = (s <= 0)
    if bad.any():
        x[bad] = 1.0 / x.shape[1]

    return _normalize_rows(x)

def maybe_filter_singlepad(df: pd.DataFrame) -> pd.DataFrame:
    if "part_name" in df.columns:
        s = df["part_name"].astype(str).str.strip().str.lower()
        return df[s == "singlepad"].copy()
    return df

def split_black_by_version_prefix(df: pd.DataFrame) -> pd.DataFrame:
    if "version_folder" not in df.columns:
        return df.copy()
    s = df["version_folder"].astype(str).str.strip().str.lower()
    return df[s.str.startswith("black")].copy()

def drop_soft_cold_weld_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if OPTIONAL_SOFT_COLD not in df.columns:
        return df
    needed = SOFT_COLS + [OPTIONAL_SOFT_COLD]
    if any(c not in df.columns for c in needed):
        return df
    soft4 = df[SOFT_COLS + [OPTIONAL_SOFT_COLD]].to_numpy(np.float64)
    soft4 = ensure_probs_like(soft4, name="soft4_for_dropcheck")
    y4 = soft4.argmax(axis=1)
    drop_mask = (y4 == 3)
    drop_n = int(drop_mask.sum())
    if drop_n > 0:
        print(f"[WARN] found {drop_n} samples where soft-argmax is COLD_WELD, drop them for 3-class eval.")
        df = df.loc[~drop_mask].copy()
    return df

def _transform_image_wrapper(img_path: str, h: int, w: int):
    """兼容 TransformImage 两种构造签名"""
    try:
        return TransformImage(img_path, rs_img_size_h=h, rs_img_size_w=w).transform()
    except TypeError:
        return TransformImage(img_path=img_path, rs_img_size_h=h, rs_img_size_w=w).transform()

def compute_binary_acc_05(probs: np.ndarray, y_true_3: np.ndarray) -> float:
    y_true_bin = (y_true_3 != 0).astype(np.int64)
    p_ok = probs[:, 0]
    p_def = 1.0 - p_ok
    y_pred_bin = (p_def >= 0.5).astype(np.int64)
    return float((y_pred_bin == y_true_bin).mean())


# =========================
# ECE + plots (和你第一份脚本一致的风格)
# =========================
def compute_ece_bins(confidences: np.ndarray, correctness: np.ndarray, n_bins: int):
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

    return float(ece), bin_edges, bin_conf, bin_acc, bin_counts

def plot_reliability_bars(bin_edges, bin_acc, title, out_path):
    width = bin_edges[1] - bin_edges[0]
    centers = bin_edges[:-1] + width / 2.0
    plt.figure(figsize=(5, 5))
    plt.bar(
        centers, bin_acc, width=width * 0.9, align="center",
        alpha=0.7, edgecolor="black", label="Empirical accuracy"
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

def make_bin_edges(n_bins: int):
    return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)

def make_bins_by_sorted_maxconf(max_conf: np.ndarray, bin_edges: np.ndarray):
    max_conf = np.asarray(max_conf, np.float64)
    sorted_idx = np.argsort(max_conf)
    mc_sorted = max_conf[sorted_idx]

    n_bins = len(bin_edges) - 1
    bins = []
    for b in range(n_bins):
        low, high = bin_edges[b], bin_edges[b + 1]
        if b == 0:
            mask = (mc_sorted >= low) & (mc_sorted <= high)
        else:
            mask = (mc_sorted > low) & (mc_sorted <= high)
        pos = np.where(mask)[0]
        bins.append(sorted_idx[pos])
    return bins

def bin_mean_maxconf(max_conf: np.ndarray, bins):
    max_conf = np.asarray(max_conf, np.float64)
    xs = np.full(len(bins), np.nan, dtype=np.float64)
    for b, idx in enumerate(bins):
        if idx.size == 0:
            continue
        xs[b] = float(max_conf[idx].mean())
    return xs

def plot_gap_bar(x, y, title, out_path, y_label, bar_width):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[mask], y[mask]
    if x2.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.bar(x2, y2, width=bar_width, edgecolor="black", alpha=0.7, align="center")
    plt.xlabel("Mean max_conf in bin")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_count_bar(x, counts, title, out_path, bar_width):
    x = np.asarray(x, np.float64)
    c = np.asarray(counts, np.float64)
    mask = np.isfinite(x) & np.isfinite(c)
    x2, c2 = x[mask], c[mask]
    if x2.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.bar(x2, c2, width=bar_width, edgecolor="black", alpha=0.7, align="center")
    plt.xlabel("Mean max_conf in bin")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_gap_hist(values, title, out_png, out_csv, bin_edges: np.ndarray):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return

    edges = np.asarray(bin_edges, dtype=np.float64)
    counts, edges = np.histogram(v, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = (edges[1] - edges[0]) * 0.9

    plt.figure(figsize=(6, 4))
    plt.bar(centers, counts, width=width, edgecolor="black", alpha=0.7, align="center")
    plt.xlabel("Gap interval")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(float(edges[0]), float(edges[-1]))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    cum = np.cumsum(counts)
    N = max(len(v), 1)

    pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "bin_center": centers,
        "count": counts,
        "ratio": counts / N,
        "cum_count": cum,
        "cum_ratio": cum / N,
    }).to_csv(out_csv, index=False, encoding="utf-8-sig")

def compute_vector_gap_hard(probs: np.ndarray, y_true: np.ndarray, bins, C: int):
    probs = np.asarray(probs, np.float64)
    y_true = np.asarray(y_true, np.int64)
    N = probs.shape[0]
    y_pred = probs.argmax(axis=1)

    n_bins = len(bins)
    gap_overall = np.full(n_bins, np.nan, dtype=np.float64)
    gap_classwise = np.full((C, n_bins), np.nan, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=int)

    vec_ece = 0.0
    vec_mce = 0.0

    for b in range(n_bins):
        idx = bins[b]
        if idx.size == 0:
            continue

        n_b = idx.size
        bin_counts[b] = n_b
        g_b = 0.0

        for k in range(C):
            idx_bk = idx[y_pred[idx] == k]
            if idx_bk.size == 0:
                continue
            acc = float(np.mean(y_true[idx_bk] == k))
            conf = float(np.mean(probs[idx_bk, k]))
            gap = abs(acc - conf)

            gap_classwise[k, b] = gap
            g_b += (idx_bk.size / n_b) * gap
            vec_ece += (idx_bk.size / N) * gap

        gap_overall[b] = g_b
        vec_mce = max(vec_mce, g_b)

    return gap_overall, gap_classwise, bin_counts, float(vec_ece), float(vec_mce)

def compute_soft_gap(probs: np.ndarray, soft: np.ndarray, y_true: np.ndarray, bins, C: int):
    probs = np.asarray(probs, np.float64)
    soft = np.asarray(soft, np.float64)
    y_true = np.asarray(y_true, np.int64)

    y_pred = probs.argmax(axis=1)
    max_conf = probs[np.arange(probs.shape[0]), y_pred]

    gap_abs_top1 = np.abs(max_conf - soft[np.arange(probs.shape[0]), y_pred])
    gap_l1 = np.abs(probs - soft).sum(axis=1)

    n_bins = len(bins)
    gap_bin_overall = np.full(n_bins, np.nan, dtype=np.float64)
    gap_bin_true_cls = np.full((C, n_bins), np.nan, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=int)

    soft_ece_like = 0.0
    soft_mce_like = 0.0
    N = probs.shape[0]

    for b in range(n_bins):
        idx = bins[b]
        if idx.size == 0:
            continue
        bin_counts[b] = idx.size

        g_b = float(np.mean(gap_abs_top1[idx]))
        gap_bin_overall[b] = g_b
        soft_ece_like += (idx.size / N) * g_b
        soft_mce_like = max(soft_mce_like, g_b)

        # per true class in this bin
        for k in range(C):
            idx_k = idx[y_true[idx] == k]
            if idx_k.size == 0:
                continue
            gap_bin_true_cls[k, b] = float(np.mean(gap_abs_top1[idx_k]))

    return {
        "gap_abs_top1": gap_abs_top1,
        "gap_l1": gap_l1,
        "gap_bin_overall": gap_bin_overall,
        "gap_bin_true_cls": gap_bin_true_cls,
        "bin_counts": bin_counts,
        "soft_ece_like": float(soft_ece_like),
        "soft_mce_like": float(soft_mce_like),
        "mean_abs_top1": float(np.mean(gap_abs_top1)),
        "mean_l1": float(np.mean(gap_l1)),
    }


# =========================
# Model loading (robust)
# =========================
def load_mpb3_ckpt_auto(ckpt_path, cfg: Dict, device):
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
            out[k[len(prefix):]] = v if k.startswith(prefix) else v
        # 注意：上面那句写错会把 key 搞没；这里做正确 strip
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

    model = MPB3net(
        backbone=cfg["backbone"],
        pretrained=False,
        n_class=cfg["n_class"],
        n_units=cfg["n_units"],
        output_form=cfg["output_type"],
    )

    # try strict
    try:
        model.load_state_dict(raw_sd, strict=True)
        print(f"=> loaded strict: {os.path.basename(ckpt_path)}")
        return model.to(device).eval()
    except Exception:
        pass

    # strip module.
    sd = raw_sd
    if any(k.startswith("module.") for k in sd.keys()):
        sd2 = _strip_prefix(sd, "module.")
        try:
            model.load_state_dict(sd2, strict=True)
            print(f"=> loaded strip_module strict: {os.path.basename(ckpt_path)}")
            return model.to(device).eval()
        except Exception:
            sd = sd2

    # strict=False fallback
    incompatible = model.load_state_dict(sd, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    print(f"[WARN] loaded strict=False: missing={len(missing)} unexpected={len(unexpected)}  ({os.path.basename(ckpt_path)})")
    return model.to(device).eval()

def load_models_from_list(ckpt_paths: List[str], cfg: Dict) -> List[MPB3net]:
    valid = []
    for p in ckpt_paths:
        if os.path.exists(p):
            valid.append(p)
        else:
            print(f"[WARN] ckpt not found, skip: {p}")
    if len(valid) == 0:
        raise ValueError("No valid ckpt paths found.")

    models = []
    for p in valid:
        print(f"\n=> load model: {p}")
        m = load_mpb3_ckpt_auto(p, cfg, DEVICE)
        models.append(m)
    print(f"\n✓ Loaded {len(models)} models on {DEVICE}")
    return models


# =========================
# Dataset loaders (统一成 soft_* / pred_* 口径)
# =========================
def _ensure_required_cols(df: pd.DataFrame):
    for c in ["ref_image", "insp_image"]:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
    if "version_folder" not in df.columns:
        df["version_folder"] = "NA"
    return df

def load_dataset_softlabel_black(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = maybe_filter_singlepad(df)
    df = split_black_by_version_prefix(df)
    df = df.reset_index(drop=True)
    df = _ensure_required_cols(df)

    # soft cols 必须存在
    missing = [c for c in SOFT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[softlabel_black] missing soft cols: {missing}")

    # drop cold_weld argmax if exists
    df = drop_soft_cold_weld_if_needed(df).reset_index(drop=True)

    return df

def load_dataset_checked_samples(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = maybe_filter_singlepad(df)
    df = df.reset_index(drop=True)
    df = _ensure_required_cols(df)

    # 软标签列：优先 soft_*；否则用 NONE_CONF 等映射
    if all(c in df.columns for c in SOFT_COLS):
        pass
    elif all(c in df.columns for c in ["NONE_CONF", "INSUFFICIENT_SOLDER_CONF", "PSEUDO_SOLDER_CONF"]):
        df["soft_ok"] = df["NONE_CONF"]
        df["soft_INSUFFICIENT_SOLDER"] = df["INSUFFICIENT_SOLDER_CONF"]
        df["soft_PSEUDO_SOLDER"] = df["PSEUDO_SOLDER_CONF"]
    else:
        raise ValueError(
            "[checked_samples] cannot find soft labels. Need either "
            "soft_ok/soft_INSUFFICIENT_SOLDER/soft_PSEUDO_SOLDER or "
            "NONE_CONF/INSUFFICIENT_SOLDER_CONF/PSEUDO_SOLDER_CONF"
        )

    return df


# =========================
# Inference
# =========================
@torch.no_grad()
def infer_model_probs_on_df(models: List[MPB3net], df: pd.DataFrame, cfg: Dict, batch_size: int) -> np.ndarray:
    """
    预计算每个模型的概率输出:
      return: model_probs [M, N, C]
    """
    N = len(df)
    M = len(models)
    C = cfg["n_class"]
    out = np.zeros((M, N, C), dtype=np.float32)

    with tqdm(total=N, desc="Infer per-model probs", unit="img") as pbar:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = df.iloc[start:end]

            img1_list, img2_list = [], []
            for _, row in batch.iterrows():
                ref_path = os.path.join(ROOT_DIR, str(row["ref_image"]))
                insp_path = os.path.join(ROOT_DIR, str(row["insp_image"]))

                ref_img = _transform_image_wrapper(ref_path, cfg["img_h"], cfg["img_w"])
                insp_img = _transform_image_wrapper(insp_path, cfg["img_h"], cfg["img_w"])

                img1_list.append(torch.FloatTensor(ref_img))
                img2_list.append(torch.FloatTensor(insp_img))

            x1 = torch.cat(img1_list, dim=0).to(DEVICE)
            x2 = torch.cat(img2_list, dim=0).to(DEVICE)

            for mi, m in enumerate(models):
                _, logits_bom = m(x1, x2)
                prob = F.softmax(logits_bom, dim=-1).detach().cpu().numpy().astype(np.float32)  # [B,C]
                out[mi, start:end, :] = prob

            pbar.update(end - start)

    return out

def combine_probs_weighted(model_probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    # [M,N,C] -> [N,C]
    return np.tensordot(w, model_probs, axes=(0, 0)).astype(np.float64)

def combine_probs_mean(model_probs: np.ndarray) -> np.ndarray:
    return model_probs.mean(axis=0).astype(np.float64)

def combine_probs_std(model_probs: np.ndarray) -> np.ndarray:
    return model_probs.std(axis=0).astype(np.float64)

def build_soup_state_dict(models: List[MPB3net], weights: np.ndarray, bn_strategy: str) -> Dict[str, torch.Tensor]:
    """
    参数 soup：按权重平均所有浮点参数。
    BN running_mean/running_var:
      - bn_strategy="first": 用第一个模型
      - bn_strategy="weighted": 也按权重平均
    num_batches_tracked：始终用第一个（整型 buffer）
    """
    assert bn_strategy in ["first", "weighted"]
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)

    sds = [m.state_dict() for m in models]
    base = sds[0]

    def _is_bn_running_stat(name: str) -> bool:
        return name.endswith("running_mean") or name.endswith("running_var")

    def _is_num_batches_tracked(name: str) -> bool:
        return name.endswith("num_batches_tracked")

    out = {}
    for name, base_param in base.items():
        if _is_num_batches_tracked(name):
            out[name] = base_param.clone()
            continue

        if _is_bn_running_stat(name):
            if bn_strategy == "first":
                out[name] = base_param.clone()
            else:
                if not base_param.dtype.is_floating_point:
                    out[name] = base_param.clone()
                else:
                    acc = torch.zeros_like(base_param)
                    for i, sd in enumerate(sds):
                        acc += float(w[i]) * sd[name]
                    out[name] = acc
            continue

        if not base_param.dtype.is_floating_point:
            out[name] = base_param.clone()
            continue

        acc = torch.zeros_like(base_param)
        for i, sd in enumerate(sds):
            acc += float(w[i]) * sd[name]
        out[name] = acc

    return out

@torch.no_grad()
def infer_single_model_on_df(model: MPB3net, df: pd.DataFrame, cfg: Dict, batch_size: int) -> np.ndarray:
    N = len(df)
    C = cfg["n_class"]
    out = np.zeros((N, C), dtype=np.float32)

    with tqdm(total=N, desc="Infer single model", unit="img") as pbar:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = df.iloc[start:end]

            img1_list, img2_list = [], []
            for _, row in batch.iterrows():
                ref_path = os.path.join(ROOT_DIR, str(row["ref_image"]))
                insp_path = os.path.join(ROOT_DIR, str(row["insp_image"]))

                ref_img = _transform_image_wrapper(ref_path, cfg["img_h"], cfg["img_w"])
                insp_img = _transform_image_wrapper(insp_path, cfg["img_h"], cfg["img_w"])

                img1_list.append(torch.FloatTensor(ref_img))
                img2_list.append(torch.FloatTensor(insp_img))

            x1 = torch.cat(img1_list, dim=0).to(DEVICE)
            x2 = torch.cat(img2_list, dim=0).to(DEVICE)

            _, logits_bom = model(x1, x2)
            prob = F.softmax(logits_bom, dim=-1).detach().cpu().numpy().astype(np.float32)
            out[start:end, :] = prob

            pbar.update(end - start)

    return out.astype(np.float64)


# =========================
# Softlabel evaluation (统一口径)
# =========================
def eval_softlabel_split(df_in: pd.DataFrame, split_name: str, out_dir: str):
    ensure_dir(out_dir)
    C = len(CLASS_ORDER)

    df = df_in.copy()
    df = _ensure_required_cols(df)

    required = ["version_folder", "ref_image", "insp_image"] + PROB_COLS + SOFT_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{split_name}] missing columns: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) == 0:
        print(f"[WARN] split={split_name} empty, skip.")
        return

    probs = ensure_probs_like(df[PROB_COLS].to_numpy(np.float64), name=f"{split_name}_probs")
    soft  = ensure_probs_like(df[SOFT_COLS].to_numpy(np.float64), name=f"{split_name}_soft")

    N = probs.shape[0]
    y_true = soft.argmax(axis=1).astype(np.int64)
    y_pred = probs.argmax(axis=1).astype(np.int64)
    max_conf = probs[np.arange(N), y_pred]
    is_correct = (y_pred == y_true).astype(np.int64)

    # Binary ok vs defect
    y_true_bin = (y_true != 0).astype(np.int64)
    p_ok = probs[:, 0]
    p_defect = probs[:, 1:].sum(axis=1)
    y_pred_bin = (p_defect >= 0.5).astype(np.int64)

    # save labeled csv
    out_df = df.copy()
    out_df["C_used"] = C
    out_df["y_true"] = y_true
    out_df["y_pred"] = y_pred
    out_df["y_true_name"] = [CLASS_ORDER[i] for i in y_true]
    out_df["y_pred_name"] = [CLASS_ORDER[i] for i in y_pred]
    out_df["max_conf"] = max_conf
    out_df["is_correct"] = is_correct
    out_df["y_true_bin"] = y_true_bin
    out_df["y_pred_bin"] = y_pred_bin
    out_df["p_ok"] = p_ok
    out_df["p_defect"] = p_defect
    labeled_csv = os.path.join(out_dir, f"{split_name}_with_labels_C{C}.csv")
    out_df.to_csv(labeled_csv, index=False, encoding="utf-8-sig")

    summary = []
    summary.append(f"[SPLIT] {split_name}  N={N}  C={C}  classes={CLASS_ORDER} (Soft Label)")

    # Binary reliability bars
    conf_bin_pred = np.where(y_pred_bin == 1, p_defect, p_ok)
    correct_bin = (y_pred_bin == y_true_bin).astype(np.int64)

    ece_bin_overall, edges_b, _, acc_b, _ = compute_ece_bins(conf_bin_pred, correct_bin, n_bins=N_BINS_ECE)
    plot_reliability_bars(
        edges_b, acc_b,
        f"[{split_name}] Binary reliability overall (top1, Soft Label)",
        os.path.join(out_dir, f"{split_name}_binary_reliability_overall.png")
    )

    ece_true_ok, edges_ok, _, acc_ok, _ = compute_ece_bins(p_ok, (y_true_bin == 0).astype(np.int64), n_bins=N_BINS_ECE)
    plot_reliability_bars(
        edges_ok, acc_ok,
        f"[{split_name}] Binary reliability (true=ok, Soft Label)",
        os.path.join(out_dir, f"{split_name}_binary_reliability_true_ok.png")
    )

    ece_true_def, edges_def, _, acc_def, _ = compute_ece_bins(p_defect, (y_true_bin == 1).astype(np.int64), n_bins=N_BINS_ECE)
    plot_reliability_bars(
        edges_def, acc_def,
        f"[{split_name}] Binary reliability (true=defect, Soft Label)",
        os.path.join(out_dir, f"{split_name}_binary_reliability_true_defect.png")
    )

    summary.append(f"  Binary overall ECE (top1) = {ece_bin_overall:.6f}")
    summary.append(f"  Binary classwise ECE (prob-wise): ok={ece_true_ok:.6f}, defect={ece_true_def:.6f}")

    # Multiclass reliability bars
    ece_m_overall, edges_m, _, acc_m, _ = compute_ece_bins(max_conf, is_correct, n_bins=N_BINS_ECE)
    plot_reliability_bars(
        edges_m, acc_m,
        f"[{split_name}] Multiclass reliability overall (top1, Soft Label)",
        os.path.join(out_dir, f"{split_name}_mclass_reliability_overall.png")
    )
    summary.append(f"  Multiclass overall ECE (top1) = {ece_m_overall:.6f}")

    summary.append("  Multiclass classwise ECE (true-class prob-wise):")
    for c, cname in enumerate(CLASS_ORDER):
        ece_c, edges_c, _, acc_c, _ = compute_ece_bins(probs[:, c], (y_true == c).astype(np.int64), n_bins=N_BINS_ECE)
        plot_reliability_bars(
            edges_c, acc_c,
            f"[{split_name}] Multiclass reliability (true={cname}, Soft Label)",
            os.path.join(out_dir, f"{split_name}_mclass_reliability_true_{cname}.png")
        )
        summary.append(f"    {cname:>18s}: {ece_c:.6f}")

    # Unified bins on max_conf (equal-width)
    bin_edges = make_bin_edges(N_BINS_ECE)
    bins = make_bins_by_sorted_maxconf(max_conf, bin_edges)

    bar_width = float(bin_edges[1] - bin_edges[0]) * 0.9
    bin_mean_mc = bin_mean_maxconf(max_conf, bins)
    bin_counts_global = np.array([len(idx) for idx in bins], dtype=int)

    nonempty = bin_counts_global > 0
    x_plot = bin_mean_mc[nonempty]
    counts_plot = bin_counts_global[nonempty]

    plot_count_bar(
        x_plot, counts_plot,
        title=f"[{split_name}] Bin counts (max_conf bins, Soft Label)",
        out_path=os.path.join(out_dir, f"{split_name}_bin_count_bar.png"),
        bar_width=bar_width
    )

    # Vector gap bars (hard)
    vec_gap_overall, vec_gap_classwise, _, vec_ece, vec_mce = compute_vector_gap_hard(
        probs, y_true, bins, C
    )
    summary.append(f"  Vector-ECE (hard) = {vec_ece:.6f}")
    summary.append(f"  Vector-MCE (hard) = {vec_mce:.6f}")

    plot_gap_bar(
        x_plot,
        vec_gap_overall[nonempty],
        title=f"[{split_name}] Vector gap per bin (hard, Soft Label)",
        out_path=os.path.join(out_dir, f"{split_name}_vector_gap_bar_overall.png"),
        y_label="Per-bin vector gap",
        bar_width=bar_width,
    )
    for k, cname in enumerate(CLASS_ORDER):
        plot_gap_bar(
            x_plot,
            vec_gap_classwise[k][nonempty],
            title=f"[{split_name}] Vector classwise gap (hard, class={cname}, Soft Label)",
            out_path=os.path.join(out_dir, f"{split_name}_vector_gap_bar_class_{cname}.png"),
            y_label="|acc - conf|",
            bar_width=bar_width,
        )

    # Soft gap bars + histogram
    soft_stats = compute_soft_gap(probs, soft, y_true, bins, C)
    summary.append(f"  Soft-ECE-like (mean gap by bins, abs top1) = {soft_stats['soft_ece_like']:.6f}")
    summary.append(f"  Soft-MCE-like (max bin mean gap, abs top1) = {soft_stats['soft_mce_like']:.6f}")
    summary.append(f"  Mean per-sample abs(top1) gap = {soft_stats['mean_abs_top1']:.6f}")
    summary.append(f"  Mean per-sample L1 gap       = {soft_stats['mean_l1']:.6f}")

    plot_gap_bar(
        x_plot,
        soft_stats["gap_bin_overall"][nonempty],
        title=f"[{split_name}] Soft gap per bin (overall, abs top1, Soft Label)",
        out_path=os.path.join(out_dir, f"{split_name}_soft_gap_bar_overall.png"),
        y_label="Mean |max_conf - soft[pred]| in bin",
        bar_width=bar_width,
    )
    for k, cname in enumerate(CLASS_ORDER):
        plot_gap_bar(
            x_plot,
            soft_stats["gap_bin_true_cls"][k][nonempty],
            title=f"[{split_name}] Soft gap per bin (true={cname}, abs top1, Soft Label)",
            out_path=os.path.join(out_dir, f"{split_name}_soft_gap_bar_true_{cname}.png"),
            y_label="Mean |max_conf - soft[pred]| in bin",
            bar_width=bar_width,
        )

    hist_edges_abs = np.linspace(0.0, 1.0, N_BINS_ECE + 1, dtype=np.float64)
    plot_gap_hist(
        soft_stats["gap_abs_top1"],
        title=f"[{split_name}] Soft ABS(top1) gap histogram (N_BINS={N_BINS_ECE}, Soft Label)",
        out_png=os.path.join(out_dir, f"{split_name}_hist_soft_abs_top1.png"),
        out_csv=os.path.join(out_dir, f"{split_name}_hist_soft_abs_top1.csv"),
        bin_edges=hist_edges_abs,
    )

    hist_edges_l1 = np.linspace(0.0, 2.0, N_BINS_ECE + 1, dtype=np.float64)
    plot_gap_hist(
        soft_stats["gap_l1"],
        title=f"[{split_name}] Soft L1 gap histogram (N_BINS={N_BINS_ECE}, Soft Label)",
        out_png=os.path.join(out_dir, f"{split_name}_hist_soft_l1.png"),
        out_csv=os.path.join(out_dir, f"{split_name}_hist_soft_l1.csv"),
        bin_edges=hist_edges_l1,
    )

    summary.append(f"  Binary acc@0.5 = {compute_binary_acc_05(probs, y_true):.6f}")

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    print("\n" + "=" * 100)
    print("\n".join(summary))
    print(f"[INFO] saved labeled csv: {labeled_csv}")
    print(f"[INFO] saved summary: {summary_path}")
    print("=" * 100 + "\n")


# =========================
# Bayesian optimization on cached probs
# =========================
def bayesian_search_on_cached_probs(model_probs: np.ndarray, soft: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    使用贝叶斯优化搜索最优权重（目标：binary acc@0.5）
    - model_probs: [M,N,C]
    - soft: [N,C] soft label
    """
    M = model_probs.shape[0]
    y_true = soft.argmax(axis=1).astype(np.int64)

    # 若没有 skopt 且没有现成 best_weights.json，这里会报错提示安装
    try:
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.utils import use_named_args
    except Exception as e:
        raise RuntimeError(
            "scikit-optimize 未安装，且 best_weights.json 不存在，无法执行贝叶斯优化。\n"
            "请安装：pip install scikit-optimize\n"
            f"原始错误: {e}"
        )

    space = [Real(0.0, 1.0, name=f"w{i}") for i in range(M)]
    best_acc = -1.0
    best_w = None

    @use_named_args(space)
    def objective(**kwargs):
        nonlocal best_acc, best_w
        w = np.array([kwargs[f"w{i}"] for i in range(M)], dtype=np.float64)
        if w.sum() <= 0:
            return 1.0
        w = w / (w.sum() + 1e-12)
        probs = combine_probs_weighted(model_probs, w)
        acc = compute_binary_acc_05(probs, y_true)
        if acc > best_acc:
            best_acc = acc
            best_w = w.copy()
            print(f"✓ New best: binary_acc@0.5={acc:.6f}, w={w}")
        return -acc

    print("\n" + "=" * 80)
    print(f"[Bayesian Search] M={M}  N_CALLS={N_CALLS}  target=binary_acc@0.5")
    print("=" * 80 + "\n")

    _ = gp_minimize(
        objective,
        space,
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL_POINTS,
        random_state=42,
        verbose=False,
    )

    return best_w, float(best_acc)


# =========================
# Main
# =========================
def main():
    ensure_dir(SAVE_DIR)

    # 1) Load models
    models = load_models_from_list(ENSEMBLE_CKPT_PATHS, CFG)
    M = len(models)

    # 2) Load two datasets
    datasets = []

    # Dataset 1: softlabel_black (black + singlepad)
    if os.path.exists(CSV_SOFTLABEL):
        df1 = load_dataset_softlabel_black(CSV_SOFTLABEL)
        datasets.append(("dataset_softlabel_black", df1))
        print(f"[DATA] dataset_softlabel_black: {len(df1)} rows from {CSV_SOFTLABEL}")
    else:
        print(f"[WARN] CSV_SOFTLABEL not found: {CSV_SOFTLABEL}")

    # Dataset 2: checked_samples (singlepad)
    if os.path.exists(CSV_CHECKED):
        df2 = load_dataset_checked_samples(CSV_CHECKED)
        datasets.append(("dataset_checked_samples", df2))
        print(f"[DATA] dataset_checked_samples: {len(df2)} rows from {CSV_CHECKED}")
    else:
        print(f"[WARN] CSV_CHECKED not found: {CSV_CHECKED}")

    if len(datasets) == 0:
        raise RuntimeError("No datasets found. Check CSV paths.")

    # 3) Precompute per-model probs for each dataset (缓存到内存；也可自行保存 .npy)
    cached = {}  # name -> dict(model_probs, soft)
    for dname, df in datasets:
        # soft labels
        soft = ensure_probs_like(df[SOFT_COLS].to_numpy(np.float64), name=f"{dname}_soft")
        model_probs = infer_model_probs_on_df(models, df, CFG, batch_size=BATCH_SIZE)  # [M,N,C]
        cached[dname] = dict(df=df, soft=soft, model_probs=model_probs)

    # 4) Get / search best weights (优先复用 best_weights.json)
    best_weights = None
    best_acc = None

    if os.path.exists(BEST_WEIGHTS_JSON):
        with open(BEST_WEIGHTS_JSON, "r", encoding="utf-8") as f:
            obj = json.load(f)
        w = np.asarray(obj.get("best_weights", []), dtype=np.float64)
        if w.size == M:
            best_weights = w
            best_acc = float(obj.get("best_binary_acc", -1.0))
            print(f"[WEIGHTS] loaded from {BEST_WEIGHTS_JSON}: acc={best_acc:.6f}, w={best_weights}")
        else:
            print(f"[WARN] best_weights.json exists but length mismatch: got {w.size}, need {M}. will re-search.")
            best_weights = None

    if best_weights is None:
        # 默认用 checked_samples 做优化；如果没有，就用 softlabel_black
        opt_name = "dataset_checked_samples" if "dataset_checked_samples" in cached else list(cached.keys())[0]
        print(f"[WEIGHTS] best_weights.json not usable. Run Bayesian search on: {opt_name}")
        model_probs_opt = cached[opt_name]["model_probs"]
        soft_opt = cached[opt_name]["soft"]

        best_weights, best_acc = bayesian_search_on_cached_probs(model_probs_opt, soft_opt)

        with open(BEST_WEIGHTS_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_weights": best_weights.tolist(),
                    "best_binary_acc": float(best_acc),
                    "optimize_on": opt_name,
                    "n_calls": N_CALLS,
                    "n_initial_points": N_INITIAL_POINTS,
                    "n_models": M,
                    "checkpoints": [os.path.basename(p) for p in ENSEMBLE_CKPT_PATHS if os.path.exists(p)],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[WEIGHTS] saved: {BEST_WEIGHTS_JSON}")

    # 5) Build two soup models (BNfirst vs BNwavg) —— 不刷 BN
    soup_first = MPB3net(
        backbone=CFG["backbone"], pretrained=False, n_class=CFG["n_class"],
        n_units=CFG["n_units"], output_form=CFG["output_type"]
    ).to(DEVICE).eval()

    soup_wavg = MPB3net(
        backbone=CFG["backbone"], pretrained=False, n_class=CFG["n_class"],
        n_units=CFG["n_units"], output_form=CFG["output_type"]
    ).to(DEVICE).eval()

    sd_first = build_soup_state_dict(models, best_weights, bn_strategy="first")
    sd_wavg  = build_soup_state_dict(models, best_weights, bn_strategy="weighted")
    soup_first.load_state_dict(sd_first, strict=True)
    soup_wavg.load_state_dict(sd_wavg, strict=True)

    # 6) Evaluate on BOTH datasets using the SAME eval_softlabel_split()
    for dname, pack in cached.items():
        df = pack["df"].copy()
        soft = pack["soft"]
        model_probs = pack["model_probs"]

        dataset_dir = os.path.join(SAVE_DIR, dname)
        ensure_dir(dataset_dir)

        # ---- A) BayesianWeightedEnsemble (prob-level weighted)
        probs_bayes = combine_probs_weighted(model_probs, best_weights)
        outA = df.copy()
        outA[PROB_COLS[0]] = probs_bayes[:, 0]
        outA[PROB_COLS[1]] = probs_bayes[:, 1]
        outA[PROB_COLS[2]] = probs_bayes[:, 2]

        # 额外保存 std（不影响测评；方便看不确定性）
        stdA = combine_probs_std(model_probs)
        outA[PROB_COLS[0] + "_STD"] = stdA[:, 0]
        outA[PROB_COLS[1] + "_STD"] = stdA[:, 1]
        outA[PROB_COLS[2] + "_STD"] = stdA[:, 2]

        runA = os.path.join(dataset_dir, "A_BayesianWeightedEnsemble")
        ensure_dir(runA)
        outA.to_csv(os.path.join(runA, "predictions.csv"), index=False, encoding="utf-8-sig")
        eval_softlabel_split(outA, split_name=dname, out_dir=runA)

        # ---- B) Soup_BNfirst
        probs_first = infer_single_model_on_df(soup_first, df, CFG, batch_size=BATCH_SIZE)
        outB = df.copy()
        outB[PROB_COLS[0]] = probs_first[:, 0]
        outB[PROB_COLS[1]] = probs_first[:, 1]
        outB[PROB_COLS[2]] = probs_first[:, 2]
        runB = os.path.join(dataset_dir, "B_Soup_BNfirst")
        ensure_dir(runB)
        outB.to_csv(os.path.join(runB, "predictions.csv"), index=False, encoding="utf-8-sig")
        eval_softlabel_split(outB, split_name=dname, out_dir=runB)

        # ---- C) Soup_BNwavg
        probs_wavg = infer_single_model_on_df(soup_wavg, df, CFG, batch_size=BATCH_SIZE)
        outC = df.copy()
        outC[PROB_COLS[0]] = probs_wavg[:, 0]
        outC[PROB_COLS[1]] = probs_wavg[:, 1]
        outC[PROB_COLS[2]] = probs_wavg[:, 2]
        runC = os.path.join(dataset_dir, "C_Soup_BNwavg")
        ensure_dir(runC)
        outC.to_csv(os.path.join(runC, "predictions.csv"), index=False, encoding="utf-8-sig")
        eval_softlabel_split(outC, split_name=dname, out_dir=runC)

        # 汇总对比（binary acc@0.5）
        y_true = soft.argmax(axis=1).astype(np.int64)
        accA = compute_binary_acc_05(probs_bayes, y_true)
        accB = compute_binary_acc_05(probs_first, y_true)
        accC = compute_binary_acc_05(probs_wavg, y_true)

        with open(os.path.join(dataset_dir, "compare_binary_acc.txt"), "w", encoding="utf-8") as f:
            f.write(f"Dataset: {dname}\n")
            f.write(f"M={M}\n")
            f.write(f"Best weights (sum=1): {best_weights.tolist()}\n\n")
            f.write(f"A_BayesianWeightedEnsemble  binary_acc@0.5 = {accA:.6f}\n")
            f.write(f"B_Soup_BNfirst             binary_acc@0.5 = {accB:.6f}\n")
            f.write(f"C_Soup_BNwavg              binary_acc@0.5 = {accC:.6f}\n")
            f.write(f"abs(B-C) = {abs(accB-accC):.6f}\n")
            f.write(f"abs(A-B) = {abs(accA-accB):.6f}\n")
            f.write(f"abs(A-C) = {abs(accA-accC):.6f}\n")

        print("\n" + "#" * 100)
        print(f"[COMPARE] {dname}")
        print(f"  A BayesianWeightedEnsemble  binary_acc@0.5 = {accA:.6f}")
        print(f"  B Soup BN=first             binary_acc@0.5 = {accB:.6f}")
        print(f"  C Soup BN=weighted          binary_acc@0.5 = {accC:.6f}")
        print(f"  abs(B-C) = {abs(accB-accC):.6f}")
        print("#" * 100 + "\n")

    print(f"[OK] Done. Outputs in: {SAVE_DIR}")


if __name__ == "__main__":
    main()
