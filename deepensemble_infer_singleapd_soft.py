#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional
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
# CONFIG - 手动配置CKPT列表
# =========================
ROOT_DIR = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310"
CSV_PATH = os.path.join(ROOT_DIR, "calibrate_with_softlabels.csv")

SAVE_DIR = os.path.join(os.path.dirname(CSV_PATH), "ensemble_eval_soft_label_black_only")
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

# =========================
# 核心修改：手动配置需要集成的CKPT列表
# =========================
# 在这里手动添加你想要参与集成的模型路径
# 格式：["路径1", "路径2", "路径3"...]，可以添加任意数量的模型
ENSEMBLE_CKPT_PATHS = [
    # # 示例路径（请替换为你实际的CKPT路径）
    # "/home/cat/workspace/vlm/scripts/models/checkpoints/model1.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2bestacc.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2bestbiacc.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2bestmlacc.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2last.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2swa.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2top0.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2top1.pth.tar",
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2top2.pth.tar",

]

BATCH_SIZE = 64
IMG_H, IMG_W = 64, 64  # singlepad

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
    兜底：clip + 归一化，保证每行是概率分布（适配Soft Label）
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
    black 只算 3 类，但如果 soft_COLD_WELD 存在且 argmax 是 cold_weld，则剔除（适配Soft Label）
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
# ECE (bins + plotting)
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
# Unified bins
# =========================
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
    """计算每个bin的平均max_conf（用于绘图）"""
    max_conf = np.asarray(max_conf, np.float64)
    xs = np.full(len(bins), np.nan, dtype=np.float64)
    for b, idx in enumerate(bins):
        if idx.size == 0:
            continue
        xs[b] = float(max_conf[idx].mean())
    return xs


# =========================
# Plot: BAR only
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
# Soft gap histogram
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
# Vector gap (hard)
# =========================
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


# =========================
# Soft gap（适配Soft Label）
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
# Model loading utilities - 适配手动配置CKPT
# =========================
def load_model_by_values(ckp, model):
    """按 value 对齐加载，兼容名字不匹配的 ckpt"""
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


def _build_ensemble_models_manual(
    ckpt_paths: List[str],
    backbone_arch: str,
    n_class: int,
    n_units: List[int],
    output_type: str,
) -> List[MPB3net]:
    """
    核心修改：从手动配置的CKPT列表构建ensemble模型
    """
    # 验证CKPT路径有效性
    valid_ckpt_paths = []
    for idx, ckpt_path in enumerate(ckpt_paths):
        if not os.path.exists(ckpt_path):
            print(f"[WARN] 第 {idx+1} 个CKPT路径不存在，跳过: {ckpt_path}")
            continue
        valid_ckpt_paths.append(ckpt_path)
    
    if len(valid_ckpt_paths) == 0:
        raise ValueError("没有有效的CKPT路径，请检查ENSEMBLE_CKPT_PATHS配置")
    
    print(f"[INFO] 共配置 {len(ckpt_paths)} 个CKPT，其中有效路径 {len(valid_ckpt_paths)} 个:")
    for i, path in enumerate(valid_ckpt_paths):
        print(f"   [{i+1}] {path}")

    # 加载每个有效模型
    models: List[MPB3net] = []
    for ckpt_path in valid_ckpt_paths:
        print(f"=> 加载 ensemble 成员: {ckpt_path}")

        model = MPB3net(
            backbone=backbone_arch,
            pretrained=False,
            n_class=n_class,
            n_units=n_units,
            output_form=output_type,
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        # 尝试提取 state_dict
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "swa_state_dict" in ckpt:
            state_dict = ckpt["swa_state_dict"]
        else:
            state_dict = ckpt

        # 处理 n_averaged 键（如果存在）
        if isinstance(state_dict, dict) and "n_averaged" in state_dict:
            state_dict = {k: v for k, v in state_dict.items() if k != "n_averaged"}

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"[WARN] strict=True 加载失败，尝试按 value 对齐: {e}")
            load_model_by_values(state_dict, model)

        model.to(DEVICE)
        model.eval()  # ensemble 保持 eval
        models.append(model)

    print(f"=> ensemble 共 {len(models)} 个有效模型，已加载到 {DEVICE}")
    return models


# =========================
# Deep Ensemble inference（适配Soft Label）
# =========================
@torch.no_grad()
def ensemble_predict_probs(
    models: List[torch.nn.Module],
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Deep Ensemble 推理：
    - 对每个模型做一次前向，取 softmax 后堆叠
    - 对模型维度求均值 / 标准差（适配Soft Label的概率分布）
    """
    bom_list = []

    for m in models:
        m.eval()
        _, logits_bom = m(x1, x2)
        bom_list.append(F.softmax(logits_bom, dim=-1))  # [B, C]

    bom_arr = torch.stack(bom_list, dim=0)  # [M, B, C]

    return {
        "bom_mean": bom_arr.mean(0),  # 集成均值（最终预测概率）
        "bom_std": bom_arr.std(0),    # 模型间标准差（不确定性）
    }


def infer_black_ensemble(
    df_black: pd.DataFrame,
    models: List[nn.Module],
    batch_size: int,
):
    """对 black 数据做 Deep Ensemble 推理（适配Soft Label数据集）"""
    N = len(df_black)
    means, stds = [], []

    with tqdm(total=N, desc="Ensemble infer (black, Soft Label)", unit="img") as pbar:
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

            if len(img1_list) == 0:
                pbar.update(end - start)
                continue

            x1 = torch.cat(img1_list, dim=0).to(DEVICE)
            x2 = torch.cat(img2_list, dim=0).to(DEVICE)

            try:
                ens_out = ensemble_predict_probs(models, x1, x2)
                m = ens_out["bom_mean"]
                s = ens_out["bom_std"]
                
                means.append(m.cpu().numpy())
                stds.append(s.cpu().numpy())
            except Exception as e:
                print(f"\n[ERROR] Batch inference failed (start={start}, end={end}): {e}")
                # 降级策略：使用第一个模型预测
                m = models[0](x1, x2)[1]
                m = F.softmax(m, dim=-1).cpu().numpy()
                s = np.zeros_like(m)
                means.append(m)
                stds.append(s)

            pbar.update(end - start)

    if len(means) == 0:
        raise ValueError("No valid predictions generated")

    mean3 = np.concatenate(means, axis=0)
    std3 = np.concatenate(stds, axis=0)
    return mean3, std3


# =========================
# Evaluation (black only) - 适配Soft Label
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
    y_true = soft.argmax(axis=1).astype(np.int64)  # Soft Label的硬标签（用于对比）
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
    summary.append(f"[SPLIT] {split_name}  N={N}  C={C}  classes={class_order} (Soft Label)")

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
    for c, cname in enumerate(class_order):
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
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_mean_mc = bin_mean_maxconf(max_conf, bins)

    bin_counts_global = np.array([len(idx) for idx in bins], dtype=int)

    # 只画非空 bin
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
    vec_gap_overall, vec_gap_classwise, vec_bin_counts, vec_ece, vec_mce = compute_vector_gap_hard(
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
    for k, cname in enumerate(class_order):
        plot_gap_bar(
            x_plot,
            vec_gap_classwise[k][nonempty],
            title=f"[{split_name}] Vector classwise gap (hard, class={cname}, Soft Label)",
            out_path=os.path.join(out_dir, f"{split_name}_vector_gap_bar_class_{cname}.png"),
            y_label="|acc - conf|",
            bar_width=bar_width,
        )

    # Soft gap bars + count (abs_top1 + l1)
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
    for k, cname in enumerate(class_order):
        plot_gap_bar(
            x_plot,
            soft_stats["gap_bin_true_cls"][k][nonempty],
            title=f"[{split_name}] Soft gap per bin (true={cname}, abs top1, Soft Label)",
            out_path=os.path.join(out_dir, f"{split_name}_soft_gap_bar_true_{cname}.png"),
            y_label="Mean |max_conf - soft[pred]| in bin",
            bar_width=bar_width,
        )

    # Soft gap histogram
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
# Main: 完整流程（手动配置CKPT + Soft Label）
# =========================
def main():
    ensure_dir(SAVE_DIR)
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] loaded Soft Label csv: {CSV_PATH}")
    print(f"[INFO] total rows: {len(df)}")

    # 只看 singlepad（若存在）
    df = maybe_filter_singlepad(df)
    print(f"[INFO] after optional part_name=singlepad filter: {len(df)}")

    # black only
    df_black = split_black_by_version_prefix(df).reset_index(drop=True)
    print(f"[INFO] black rows (version_folder startswith 'black'): {len(df_black)}")
    if len(df_black) == 0:
        print("[WARN] black split is empty. Check version_folder prefix rule.")
        return

    # 如果黑数据里存在 soft_COLD_WELD 且实际是最大类，直接剔除
    df_black = drop_soft_cold_weld_if_needed(df_black).reset_index(drop=True)
    print(f"[INFO] black rows after drop-soft-cold-weld(if any): {len(df_black)}")

    # 必要列检查
    for c in ["ref_image", "insp_image", "version_folder"]:
        if c not in df_black.columns:
            raise ValueError(f"Soft Label CSV missing required column: {c}")

    # 确保 prob 列存在（要覆盖写回）
    for c in BLACK_PROB_COLS:
        if c not in df_black.columns:
            df_black[c] = np.nan
    if OPTIONAL_PROB_COLD not in df_black.columns:
        df_black[OPTIONAL_PROB_COLD] = 0.0

    # 核心修改：从手动配置的CKPT列表构建Ensemble模型组
    models = _build_ensemble_models_manual(
        ckpt_paths=ENSEMBLE_CKPT_PATHS,
        backbone_arch="fcdropoutmobilenetv3large",
        n_class=3,
        n_units=[256, 256],
        output_type="dual2",
    )

    # Ensemble 推理配置
    tag = f"ensemble_{len(models)}models_manual_ckpt_soft_label"
    run_dir = os.path.join(SAVE_DIR, tag)
    ensure_dir(run_dir)

    print("\n" + "=" * 90)
    print(f"[RUN {tag}] Ensemble infer black only (C=3, Soft Label)")
    print("=" * 90)

    # Ensemble 推理（适配Soft Label）
    mean3, std3 = infer_black_ensemble(df_black, models, batch_size=BATCH_SIZE)

    # 写入预测概率
    out_black = df_black.copy()
    out_black[BLACK_PROB_COLS[0]] = mean3[:, 0]
    out_black[BLACK_PROB_COLS[1]] = mean3[:, 1]
    out_black[BLACK_PROB_COLS[2]] = mean3[:, 2]
    out_black[OPTIONAL_PROB_COLD] = 0.0  # black 不考虑 cold_weld

    # 保存 STD（不确定性）
    out_black[BLACK_PROB_COLS[0] + "_STD"] = std3[:, 0]
    out_black[BLACK_PROB_COLS[1] + "_STD"] = std3[:, 1]
    out_black[BLACK_PROB_COLS[2] + "_STD"] = std3[:, 2]
    out_black[OPTIONAL_PROB_COLD + "_STD"] = 0.0

    out_csv = os.path.join(run_dir, f"black_ensemble_{tag}.csv")
    out_black.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved ensemble predictions (Soft Label): {out_csv}")

    # 完整评估（适配Soft Label）
    eval_one_split(
        out_black, "black",
        out_dir=os.path.join(run_dir, "black"),
        class_order=BLACK_CLASS_ORDER,
        prob_cols=BLACK_PROB_COLS,
        soft_cols=BLACK_SOFT_COLS,
    )

    print(f"[OK] done. Soft Label outputs in: {SAVE_DIR}")


if __name__ == "__main__":
    main()