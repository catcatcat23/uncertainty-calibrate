#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.optim.swa_utils import AveragedModel

# 你的工程依赖
from utils.utilities import TransformImage
from models.MPB3 import MPB3net


# =========================
# CONFIG
# =========================
ROOT_DIR = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310"
CSV_PATH = os.path.join(ROOT_DIR, "calibrate_with_softlabels.csv")

SAVE_DIR = os.path.join(os.path.dirname(CSV_PATH), "laplace_eval_soft_label_black_only")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 统一分箱：vector/soft 共用（等宽 bins；按 max_conf 取值分箱；实现上先 sort 再落回原 idx）
N_BINS_ECE = 15

# black 用 3 类（不考虑 COLD_WELD）
BLACK_CLASS_ORDER = ["ok", "INSUFFICIENT_SOLDER", "PSEUDO_SOLDER"]
BLACK_PROB_COLS = [
    "model_test_ok",
    "model_test_INSUFFICIENT_SOLDER",
    "model_test_PSEUDO_SOLDER",
]
BLACK_SOFT_COLS = [
    "soft_ok",
    "soft_INSUFFICIENT_SOLDER",
    "soft_PSEUDO_SOLDER",
]

OPTIONAL_SOFT_COLD = "soft_COLD_WELD"
OPTIONAL_PROB_COLD = "model_test_COLD_WELD"

# Laplace 参数
PRIOR_PRECISION = 1.0
FISHER_SCALE = 1.0
MC_SAMPLES = 20  # Laplace 采样次数
BATCH_SIZE = 64

IMG_H, IMG_W = 64, 64  # singlepad

# 你的 ckpt（3 类）
CKPT_C3 = (
    "/home/cat/workspace/vlm/scripts/models/checkpoints/singlepadfcdropoutmobilenetv3largers6464s42c3val0.1b256_ckp_v0.18.9lhf1certainlut05cp05clean20.0j0.4lr0.025nb256nm256dual2top2.pth.tar"
)

plt.rcParams["font.family"] = "DejaVu Sans"


# =========================
# Utils
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    s = mat.sum(axis=1, keepdims=True)
    return mat / (s + 1e-12)

def ensure_probs_like(mat: np.ndarray, name: str = "probs") -> np.ndarray:
    """
    兜底：clip + 归一化，保证每行是概率分布
    """
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

    # 再次严格归一化
    x = _normalize_rows(x)
    return x

def split_black_by_version_prefix(df: pd.DataFrame):
    if "version_folder" not in df.columns:
        raise ValueError("CSV missing column: version_folder")
    s = df["version_folder"].astype(str).str.strip().str.lower()
    return df[s.str.startswith("black")].copy()

def maybe_filter_singlepad(df: pd.DataFrame):
    if "part_name" in df.columns:
        s = df["part_name"].astype(str).str.strip().str.lower()
        return df[s == "singlepad"].copy()
    return df

def drop_soft_cold_weld_if_needed(df_black: pd.DataFrame) -> pd.DataFrame:
    """
    black 只算 3 类，但如果 soft_COLD_WELD 存在且 argmax 是 cold_weld，则剔除
    """
    if OPTIONAL_SOFT_COLD not in df_black.columns:
        return df_black

    needed = BLACK_SOFT_COLS + [OPTIONAL_SOFT_COLD]
    if any(c not in df_black.columns for c in needed):
        return df_black

    soft4 = df_black[BLACK_SOFT_COLS + [OPTIONAL_SOFT_COLD]].to_numpy(np.float64)
    soft4 = ensure_probs_like(soft4, name="black_soft4_for_dropcheck")
    y4 = soft4.argmax(axis=1)
    drop_mask = (y4 == 3)
    drop_n = int(drop_mask.sum())
    if drop_n > 0:
        print(f"[WARN] black: found {drop_n} samples where soft-argmax is COLD_WELD, drop them for 3-class eval.")
        df_black = df_black.loc[~drop_mask].copy()
    return df_black


# =========================
# ECE (bins + plotting) — 和你参考代码一致
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


# =========================
# Unified bins: equal-width on max_conf (实现上：sort->mask->回原 idx)
# =========================
def make_bin_edges(n_bins: int):
    return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)

def make_bins_by_sorted_maxconf(max_conf: np.ndarray, bin_edges: np.ndarray):
    """
    先按 max_conf 排序，再按等宽 bin_edges 划分。
    返回：bins_orig_indices: list[np.ndarray]，每个元素是该 bin 的"原始样本下标"
    """
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
    """
    （保留函数，不再用于柱子位置）
    """
    max_conf = np.asarray(max_conf, np.float64)
    xs = np.full(len(bins), np.nan, dtype=np.float64)
    for b, idx in enumerate(bins):
        if idx.size == 0:
            continue
        xs[b] = float(max_conf[idx].mean())
    return xs


# =========================
# Plot: BAR only (gap + count)
# =========================
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


# =========================
# Soft gap histogram (替代 CDF)：统计各 gap 区间样本数量 + CSV
# =========================
def plot_gap_hist(values, title, out_png, out_csv, bin_edges: np.ndarray):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return

    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must be 1D array with length >= 2")

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


# =========================
# Vector gap (hard): same bins, pred-grouped, gap=|acc-conf|
# =========================
def compute_vector_gap_hard(probs: np.ndarray, y_true: np.ndarray, bins, C: int):
    """
    返回：
      - gap_overall_per_bin[b] = sum_k (n_bk/n_b) * |acc_bk - conf_bk|
      - gap_classwise[k,b] = |acc_bk - conf_bk|（pred==k为空则NaN）
      - vec_ece = sum_{b,k} (n_bk/N)*|acc-conf|
      - vec_mce = max_b gap_overall_per_bin[b]
      - bin_counts[b]
    """
    probs = np.asarray(probs, np.float64)
    y_true = np.asarray(y_true, np.int64)
    N = probs.shape[0]
    y_pred = probs.argmax(axis=1)

    n_bins = len(bins)
    gap_overall = np.full(n_bins, np.nan, dtype=np.float64)
    gap_classwise = np.full((C, n_bins), np.nan, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

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


# =========================
# Soft gap: same bins, per-sample diff then mean in bin
#   - overall: mean(|max_conf - soft[pred]|)
#   - classwise (by TRUE class): bin 内按 y_true 分 3 类再取均值
# =========================
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
    bin_counts = np.zeros(n_bins, dtype=np.int64)

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
# Model loading
# =========================
def load_mpb3_ckpt_auto(ckpt_path, backbone_arch, n_class, n_units, output_type, device):
    """
    模型加载：自动适配 SWA / DP / DDP / AveragedModel 等多种 ckpt 结构
    """
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
            print(f"=> loaded ckpt ({tag}) with strict=True: {ckpt_path}")
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


# =========================
# Laplace: 找 BOM 最后一层
# =========================
def _find_bom_last_layer(model: nn.Module, n_class: int) -> nn.Linear:
    """在整个模型里搜索 out_features == n_class 的 nn.Linear，把最后一个当作 BOM 头。"""
    candidates = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == n_class:
            candidates.append(m)

    if not candidates:
        raise RuntimeError(f"no nn.Linear with out_features={n_class} found")

    if len(candidates) > 1:
        print("[WARN] multiple candidate BOM heads, use the last one:")
        for idx, layer in enumerate(candidates):
            print(
                f"   [{idx}] in_features={layer.in_features}, "
                f"out_features={layer.out_features}"
            )

    bom_layer = candidates[-1]
    print(
        f"[INFO] BOM head Linear: "
        f"in_features={bom_layer.in_features}, out_features={bom_layer.out_features}"
    )
    return bom_layer


# =========================
# Laplace: 拟合（经验 Fisher）
# =========================
def fit_laplace_last_layer_bom(
    model: nn.Module,
    df_black: pd.DataFrame,
    prior_precision: float = 1.0,
    fisher_scale: float = 1.0,
    batch_size: int = 64,
):
    """
    使用 CSV 中的软标签拟合 BOM 头最后一层 Linear 的 Laplace 后验：
    - 只对 BOM 最后一层做 Laplace（last-layer）
    - 标签来自 soft label argmax
    - 使用经验 Fisher：E[g^2] 估计 Hessian 对角
    """
    n_class = 3
    bom_layer = _find_bom_last_layer(model, n_class)

    # 只训练 BOM 最后一层
    for p in model.parameters():
        p.requires_grad_(False)
    for p in bom_layer.parameters():
        p.requires_grad_(True)

    fisher_w = torch.zeros_like(bom_layer.weight, device=DEVICE)
    fisher_b = torch.zeros_like(bom_layer.bias, device=DEVICE)
    n_batches = 0

    model.eval()

    # 从软标签获取 ground truth
    soft = ensure_probs_like(
        df_black[BLACK_SOFT_COLS].to_numpy(np.float64),
        name="black_soft_for_laplace"
    )
    y_true = soft.argmax(axis=1).astype(np.int64)

    N = len(df_black)
    print("[*] start Laplace fitting (last-layer, empirical Fisher)")
    
    with tqdm(total=N, desc="Laplace-fit", unit="img") as pbar:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = df_black.iloc[start:end]
            batch_y = y_true[start:end]

            img1_list, img2_list = [], []
            for _, row in batch.iterrows():
                ref_rel = row["ref_image"]
                insp_rel = row["insp_image"]

                ref_path = os.path.join(ROOT_DIR, ref_rel)
                insp_path = os.path.join(ROOT_DIR, insp_rel)

                ref_img = TransformImage(ref_path, rs_img_size_h=IMG_H, rs_img_size_w=IMG_W).transform()
                insp_img = TransformImage(insp_path, rs_img_size_h=IMG_H, rs_img_size_w=IMG_W).transform()

                img1_list.append(torch.FloatTensor(ref_img))
                img2_list.append(torch.FloatTensor(insp_img))

            x1 = torch.cat(img1_list, dim=0).to(DEVICE)
            x2 = torch.cat(img2_list, dim=0).to(DEVICE)
            y = torch.LongTensor(batch_y).to(DEVICE)

            _, logits_bom = model(x1, x2)
            log_probs = F.log_softmax(logits_bom, dim=-1)
            nll = F.nll_loss(log_probs, y, reduction="mean")

            model.zero_grad(set_to_none=True)
            nll.backward()

            fisher_w += bom_layer.weight.grad.detach() ** 2
            fisher_b += bom_layer.bias.grad.detach() ** 2

            n_batches += 1
            pbar.update(end - start)

    fisher_w /= max(n_batches, 1)
    fisher_b /= max(n_batches, 1)

    precision_w = prior_precision + fisher_scale * fisher_w
    precision_b = prior_precision + fisher_scale * fisher_b

    var_w = 1.0 / (precision_w + 1e-8)
    var_b = 1.0 / (precision_b + 1e-8)

    laplace_state = {
        "mean_weight": bom_layer.weight.detach().clone(),
        "mean_bias": bom_layer.bias.detach().clone(),
        "var_weight": var_w.detach().clone(),
        "var_bias": var_b.detach().clone(),
    }

    for p in model.parameters():
        p.requires_grad_(False)

    print("[*] Laplace fitting done (using CSV soft-label argmax)")
    return bom_layer, laplace_state


# =========================
# Laplace 推理：对 BOM 头做参数采样
# =========================
@torch.no_grad()
def laplace_predict_probs(
    model: nn.Module,
    bom_layer: nn.Linear,
    laplace_state: dict,
    x1: torch.Tensor,
    x2: torch.Tensor,
    mc_samples: int = 50,
):
    """使用 Laplace 后验对 (W,b) 采样，多次 forward，返回 softmax 概率的均值 & 标准差。"""
    w_mean = laplace_state["mean_weight"].to(DEVICE)
    b_mean = laplace_state["mean_bias"].to(DEVICE)
    w_std = laplace_state["var_weight"].to(DEVICE).clamp(min=1e-12).sqrt()
    b_std = laplace_state["var_bias"].to(DEVICE).clamp(min=1e-12).sqrt()

    # 还原为 mean 权重
    bom_layer.weight.data = w_mean.clone()
    bom_layer.bias.data = b_mean.clone()

    probs_list = []

    for _ in range(mc_samples):
        # 采样扰动
        eps_w = torch.randn_like(w_mean)
        eps_b = torch.randn_like(b_mean)

        sampled_w = w_mean + w_std * eps_w
        sampled_b = b_mean + b_std * eps_b

        # 临时替换参数
        bom_layer.weight.data = sampled_w
        bom_layer.bias.data = sampled_b

        _, logits_bom = model(x1, x2)
        probs_bom = F.softmax(logits_bom, dim=-1)
        probs_list.append(probs_bom)

    # 还原为 mean 权重
    bom_layer.weight.data = w_mean.clone()
    bom_layer.bias.data = b_mean.clone()

    probs_arr = torch.stack(probs_list, dim=0)  # [S, B, C]
    mean_probs = probs_arr.mean(dim=0)
    std_probs = probs_arr.std(dim=0)

    return mean_probs, std_probs


def infer_black_laplace(
    df_black: pd.DataFrame,
    model: nn.Module,
    bom_layer: nn.Linear,
    laplace_state: dict,
    mc_samples: int,
    batch_size: int,
):
    """对 black 数据做 Laplace 推理"""
    N = len(df_black)
    means, stds = [], []

    with tqdm(total=N, desc="Laplace infer (black)", unit="img") as pbar:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = df_black.iloc[start:end]

            img1_list, img2_list = [], []
            for _, row in batch.iterrows():
                ref_rel = row["ref_image"]
                insp_rel = row["insp_image"]

                ref_path = os.path.join(ROOT_DIR, ref_rel)
                insp_path = os.path.join(ROOT_DIR, insp_rel)

                ref_img = TransformImage(ref_path, rs_img_size_h=IMG_H, rs_img_size_w=IMG_W).transform()
                insp_img = TransformImage(insp_path, rs_img_size_h=IMG_H, rs_img_size_w=IMG_W).transform()

                img1_list.append(torch.FloatTensor(ref_img))
                img2_list.append(torch.FloatTensor(insp_img))

            x1 = torch.cat(img1_list, dim=0).to(DEVICE)
            x2 = torch.cat(img2_list, dim=0).to(DEVICE)

            m, s = laplace_predict_probs(
                model, bom_layer, laplace_state, x1, x2, mc_samples=mc_samples
            )
            means.append(m.cpu().numpy())
            stds.append(s.cpu().numpy())

            pbar.update(end - start)

    mean3 = np.concatenate(means, axis=0)
    std3 = np.concatenate(stds, axis=0)
    return mean3, std3


# =========================
# Evaluation (black only)
# =========================
def eval_one_split(df: pd.DataFrame, split_name: str, out_dir: str,
                   class_order, prob_cols, soft_cols):
    ensure_dir(out_dir)
    C = len(class_order)

    required = ["version_folder", "ref_image", "insp_image"] + prob_cols + soft_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{split_name}] missing columns: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) == 0:
        print(f"[WARN] split={split_name} empty, skip.")
        return

    probs = ensure_probs_like(df[prob_cols].to_numpy(np.float64), name=f"{split_name}_probs")
    soft = ensure_probs_like(df[soft_cols].to_numpy(np.float64), name=f"{split_name}_soft")

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
    out_df["y_true_name"] = [class_order[i] for i in y_true]
    out_df["y_pred_name"] = [class_order[i] for i in y_pred]
    out_df["max_conf"] = max_conf
    out_df["is_correct"] = is_correct
    out_df["y_true_bin"] = y_true_bin
    out_df["y_pred_bin"] = y_pred_bin
    out_df["p_ok"] = p_ok
    out_df["p_defect"] = p_defect
    labeled_csv = os.path.join(out_dir, f"{split_name}_with_labels_C{C}.csv")
    out_df.to_csv(labeled_csv, index=False, encoding="utf-8-sig")

    summary = []
    summary.append(f"[SPLIT] {split_name}  N={N}  C={C}  classes={class_order}")

    # =========================
    # (A) Binary reliability bars
    # =========================
    conf_bin_pred = np.where(y_pred_bin == 1, p_defect, p_ok)
    correct_bin = (y_pred_bin == y_true_bin).astype(np.int64)

    ece_bin_overall, edges_b, _, acc_b, _ = compute_ece_bins(conf_bin_pred, correct_bin, n_bins=N_BINS_ECE)
    plot_reliability_bars(
        edges_b, acc_b,
        f"[{split_name}] Binary reliability overall (top1)",
        os.path.join(out_dir, f"{split_name}_binary_reliability_overall.png")
    )

    ece_true_ok, edges_ok, _, acc_ok, _ = compute_ece_bins(p_ok, (y_true_bin == 0).astype(np.int64), n_bins=N_BINS_ECE)
    plot_reliability_bars(
        edges_ok, acc_ok,
        f"[{split_name}] Binary reliability (true=ok)",
        os.path.join(out_dir, f"{split_name}_binary_reliability_true_ok.png")
    )

    ece_true_def, edges_def, _, acc_def, _ = compute_ece_bins(p_defect, (y_true_bin == 1).astype(np.int64), n_bins=N_BINS_ECE)
    plot_reliability_bars(
        edges_def, acc_def,
        f"[{split_name}] Binary reliability (true=defect)",
        os.path.join(out_dir, f"{split_name}_binary_reliability_true_defect.png")
    )

    summary.append(f"  Binary overall ECE (top1) = {ece_bin_overall:.6f}")
    summary.append(f"  Binary classwise ECE (prob-wise): ok={ece_true_ok:.6f}, defect={ece_true_def:.6f}")

    # =========================
    # (B) Multiclass reliability bars
    # =========================
    ece_m_overall, edges_m, _, acc_m, _ = compute_ece_bins(max_conf, is_correct, n_bins=N_BINS_ECE)
    plot_reliability_bars(
        edges_m, acc_m,
        f"[{split_name}] Multiclass reliability overall (top1)",
        os.path.join(out_dir, f"{split_name}_mclass_reliability_overall.png")
    )
    summary.append(f"  Multiclass overall ECE (top1) = {ece_m_overall:.6f}")

    summary.append("  Multiclass classwise ECE (true-class prob-wise):")
    for c, cname in enumerate(class_order):
        ece_c, edges_c, _, acc_c, _ = compute_ece_bins(probs[:, c], (y_true == c).astype(np.int64), n_bins=N_BINS_ECE)
        plot_reliability_bars(
            edges_c, acc_c,
            f"[{split_name}] Multiclass reliability (true={cname})",
            os.path.join(out_dir, f"{split_name}_mclass_reliability_true_{cname}.png")
        )
        summary.append(f"    {cname:>18s}: {ece_c:.6f}")

    # =========================
    # (C) Unified bins on max_conf (equal-width)
    # =========================
    bin_edges = make_bin_edges(N_BINS_ECE)
    bins = make_bins_by_sorted_maxconf(max_conf, bin_edges)

    bar_width = float(bin_edges[1] - bin_edges[0]) * 0.9
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    bin_counts_global = np.array([len(idx) for idx in bins], dtype=np.int64)

    nonempty = bin_counts_global > 0
    x_plot = bin_centers[nonempty]
    counts_plot = bin_counts_global[nonempty]

    plot_count_bar(
        x_plot, counts_plot,
        title=f"[{split_name}] Bin counts (max_conf bins)",
        out_path=os.path.join(out_dir, f"{split_name}_bin_count_bar.png"),
        bar_width=bar_width
    )

    # =========================
    # (D) Vector gap bars (hard): overall + 3 classwise
    # =========================
    vec_gap_overall, vec_gap_classwise, vec_bin_counts, vec_ece, vec_mce = compute_vector_gap_hard(
        probs, y_true, bins, C
    )
    summary.append(f"  Vector-ECE (hard) = {vec_ece:.6f}")
    summary.append(f"  Vector-MCE (hard) = {vec_mce:.6f}")

    plot_gap_bar(
        x_plot,
        vec_gap_overall[nonempty],
        title=f"[{split_name}] Vector gap per bin (hard)",
        out_path=os.path.join(out_dir, f"{split_name}_vector_gap_bar_overall.png"),
        y_label="Per-bin vector gap",
        bar_width=bar_width,
    )
    for k, cname in enumerate(class_order):
        plot_gap_bar(
            x_plot,
            vec_gap_classwise[k][nonempty],
            title=f"[{split_name}] Vector classwise gap (hard, class={cname})",
            out_path=os.path.join(out_dir, f"{split_name}_vector_gap_bar_class_{cname}.png"),
            y_label="|acc - conf|",
            bar_width=bar_width,
        )

    # =========================
    # (E) Soft gap bars + histogram
    # =========================
    soft_stats = compute_soft_gap(probs, soft, y_true, bins, C)
    summary.append(f"  Soft-ECE-like (mean gap by bins, abs top1) = {soft_stats['soft_ece_like']:.6f}")
    summary.append(f"  Soft-MCE-like (max bin mean gap, abs top1) = {soft_stats['soft_mce_like']:.6f}")
    summary.append(f"  Mean per-sample abs(top1) gap = {soft_stats['mean_abs_top1']:.6f}")
    summary.append(f"  Mean per-sample L1 gap       = {soft_stats['mean_l1']:.6f}")

    plot_gap_bar(
        x_plot,
        soft_stats["gap_bin_overall"][nonempty],
        title=f"[{split_name}] Soft gap per bin (overall, abs top1)",
        out_path=os.path.join(out_dir, f"{split_name}_soft_gap_bar_overall.png"),
        y_label="Mean |max_conf - soft[pred]| in bin",
        bar_width=bar_width,
    )
    for k, cname in enumerate(class_order):
        plot_gap_bar(
            x_plot,
            soft_stats["gap_bin_true_cls"][k][nonempty],
            title=f"[{split_name}] Soft gap per bin (true={cname}, abs top1)",
            out_path=os.path.join(out_dir, f"{split_name}_soft_gap_bar_true_{cname}.png"),
            y_label="Mean |max_conf - soft[pred]| in bin",
            bar_width=bar_width,
        )

    # Histogram
    hist_edges_abs = np.linspace(0.0, 1.0, N_BINS_ECE + 1, dtype=np.float64)
    plot_gap_hist(
        soft_stats["gap_abs_top1"],
        title=f"[{split_name}] Soft ABS(top1) gap histogram",
        out_png=os.path.join(out_dir, f"{split_name}_hist_soft_abs_top1.png"),
        out_csv=os.path.join(out_dir, f"{split_name}_hist_soft_abs_top1.csv"),
        bin_edges=hist_edges_abs,
    )

    hist_edges_l1 = np.linspace(0.0, 2.0, N_BINS_ECE + 1, dtype=np.float64)
    plot_gap_hist(
        soft_stats["gap_l1"],
        title=f"[{split_name}] Soft L1 gap histogram",
        out_png=os.path.join(out_dir, f"{split_name}_hist_soft_l1.png"),
        out_csv=os.path.join(out_dir, f"{split_name}_hist_soft_l1.csv"),
        bin_edges=hist_edges_l1,
    )

    # summary.txt
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    print("\n" + "=" * 100)
    print("\n".join(summary))
    print(f"[INFO] saved labeled csv: {labeled_csv}")
    print(f"[INFO] saved summary: {summary_path}")
    print("=" * 100 + "\n")


# =========================
# main: black only + Laplace + eval/plot
# =========================
def main():
    ensure_dir(SAVE_DIR)
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] loaded csv: {CSV_PATH}")
    print(f"[INFO] total rows: {len(df)}")

    # 只看 singlepad
    df = maybe_filter_singlepad(df)
    print(f"[INFO] after optional part_name=singlepad filter: {len(df)}")

    # black only
    df_black = split_black_by_version_prefix(df).reset_index(drop=True)
    print(f"[INFO] black rows (version_folder startswith 'black'): {len(df_black)}")
    if len(df_black) == 0:
        print("[WARN] black split is empty. Check version_folder prefix rule.")
        return

    # 剔除 soft_COLD_WELD
    df_black = drop_soft_cold_weld_if_needed(df_black).reset_index(drop=True)
    print(f"[INFO] black rows after drop-soft-cold-weld(if any): {len(df_black)}")

    # 必要列检查
    for c in ["ref_image", "insp_image", "version_folder"]:
        if c not in df_black.columns:
            raise ValueError(f"CSV missing required column: {c}")

    # 确保 prob 列存在
    for c in BLACK_PROB_COLS:
        if c not in df_black.columns:
            df_black[c] = np.nan
    if OPTIONAL_PROB_COLD not in df_black.columns:
        df_black[OPTIONAL_PROB_COLD] = 0.0

    # 模型加载（3 类）
    if not os.path.exists(CKPT_C3):
        raise FileNotFoundError(f"ckpt not found: {CKPT_C3}")

    model = load_mpb3_ckpt_auto(
        ckpt_path=CKPT_C3,
        backbone_arch="fcdropoutmobilenetv3large",
        n_class=3,
        n_units=[256, 256],
        output_type="dual2",
        device=DEVICE,
    )

    tag = f"laplace_prior{PRIOR_PRECISION}_fisher{FISHER_SCALE}_mc{MC_SAMPLES}"
    run_dir = os.path.join(SAVE_DIR, tag)
    ensure_dir(run_dir)

    print("\n" + "=" * 90)
    print(f"[RUN {tag}] Laplace infer black only (C=3)")
    print("=" * 90)

    # 拟合 Laplace
    bom_layer, laplace_state = fit_laplace_last_layer_bom(
        model=model,
        df_black=df_black,
        prior_precision=PRIOR_PRECISION,
        fisher_scale=FISHER_SCALE,
        batch_size=BATCH_SIZE,
    )

    # 推理
    mean3, std3 = infer_black_laplace(
        df_black, model, bom_layer, laplace_state,
        mc_samples=MC_SAMPLES, batch_size=BATCH_SIZE
    )

    out_black = df_black.copy()
    out_black[BLACK_PROB_COLS[0]] = mean3[:, 0]
    out_black[BLACK_PROB_COLS[1]] = mean3[:, 1]
    out_black[BLACK_PROB_COLS[2]] = mean3[:, 2]
    out_black[OPTIONAL_PROB_COLD] = 0.0

    # 保存 STD
    out_black[BLACK_PROB_COLS[0] + "_STD"] = std3[:, 0]
    out_black[BLACK_PROB_COLS[1] + "_STD"] = std3[:, 1]
    out_black[BLACK_PROB_COLS[2] + "_STD"] = std3[:, 2]
    out_black[OPTIONAL_PROB_COLD + "_STD"] = 0.0

    out_csv = os.path.join(run_dir, f"black_laplace_overwrite_{tag}.csv")
    out_black.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved overwritten csv: {out_csv}")

    # eval + plots
    eval_one_split(
        out_black, "black",
        out_dir=os.path.join(run_dir, "black"),
        class_order=BLACK_CLASS_ORDER,
        prob_cols=BLACK_PROB_COLS,
        soft_cols=BLACK_SOFT_COLS,
    )

    print(f"[OK] done. outputs in: {SAVE_DIR}")


if __name__ == "__main__":
    main()