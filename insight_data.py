#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# CONFIG (your paths)
# =========================
ROOT_DIR = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310"
CSV_PATH = os.path.join(ROOT_DIR, "calibrate_with_softlabels.csv")

SAVE_DIR = os.path.join(os.path.dirname(CSV_PATH), "mcdropout_eval_soft_label_black_only")
os.makedirs(SAVE_DIR, exist_ok=True)

# bins (used for hist; you can change)
HIST_BINS = 50

# black 3-class soft label columns
BLACK_CLASS_ORDER = ["ok", "INSUFFICIENT_SOLDER", "PSEUDO_SOLDER"]
BLACK_SOFT_COLS = [
    "soft_ok",
    "soft_INSUFFICIENT_SOLDER",
    "soft_PSEUDO_SOLDER",
]

# Use an English-friendly default font (no SimHei)
plt.rcParams["font.family"] = "DejaVu Sans"


# =========================
# Utils
# =========================
def _safe_to_float(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _maybe_renorm_soft(soft: np.ndarray, eps: float = 1e-12):
    """
    If soft_* does not strictly sum to 1 (rounding/missing), renormalize row-wise.
    Rows with sum<=eps remain unchanged.
    """
    s = soft.sum(axis=1, keepdims=True)
    mask = (s > eps).squeeze(1)
    soft2 = soft.copy()
    soft2[mask] = soft2[mask] / s[mask]
    return soft2


def save_hist(data, bins, title, xlabel, out_path):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("[SAVE]", out_path)


def save_bar(names, values, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(6, 4))
    plt.bar(names, values, edgecolor="black", alpha=0.9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("[SAVE]", out_path)


def desc_stats(x: np.ndarray, name: str):
    x = np.asarray(x, dtype=np.float64)
    qs = np.quantile(x, [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
    print(f"\n[{name}]")
    print(f"  mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}")
    print(
        "  q01={:.4f}, q05={:.4f}, q10={:.4f}, q50={:.4f}, q90={:.4f}, q95={:.4f}, q99={:.4f}".format(
            qs[0], qs[1], qs[2], qs[3], qs[4], qs[5], qs[6]
        )
    )


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(CSV_PATH)

    # check required columns
    for c in BLACK_SOFT_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")

    df = _safe_to_float(df, BLACK_SOFT_COLS)

    soft = df[BLACK_SOFT_COLS].to_numpy(dtype=np.float64)

    # drop invalid rows
    valid_mask = np.isfinite(soft).all(axis=1)
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(f"[WARN] Found NaN/Inf in soft_* columns, dropped rows = {dropped}")
    soft = soft[valid_mask]

    # clamp and renormalize
    soft = np.clip(soft, 0.0, 1.0)
    soft = _maybe_renorm_soft(soft)

    N = soft.shape[0]
    print(f"[INFO] Valid samples N={N}")
    if N == 0:
        print("[WARN] No valid samples. Exit.")
        return

    # per-sample stats
    argmax = soft.argmax(axis=1)                  # [N]
    max_conf = soft[np.arange(N), argmax]         # [N]

    # margin = top1 - top2
    part = np.partition(soft, -2, axis=1)
    top2 = part[:, -2]
    margin = max_conf - top2

    # entropy
    eps = 1e-12
    entropy = -(soft * np.log(soft + eps)).sum(axis=1)

    # -------------------------
    # Print stats
    # -------------------------
    for i, cname in enumerate(BLACK_CLASS_ORDER):
        desc_stats(soft[:, i], f"soft_{cname}")

    desc_stats(max_conf, "max_conf = max(soft_*)")
    desc_stats(margin, "margin = top1 - top2")
    desc_stats(entropy, "entropy(soft)")

    counts = np.bincount(argmax, minlength=len(BLACK_CLASS_ORDER))
    ratios = counts / max(1, counts.sum())

    print("\n[argmax class distribution]")
    for i, cname in enumerate(BLACK_CLASS_ORDER):
        print(f"  {cname}: {counts[i]} ({ratios[i]*100:.2f}%)")

    for thr in [0.6, 0.7, 0.8, 0.9]:
        print(f"[max_conf >= {thr:.1f}] ratio = {(max_conf >= thr).mean()*100:.2f}%")

    # -------------------------
    # Plots (English only)
    # -------------------------
    # (1) per-class soft probability distributions
    for i, cname in enumerate(BLACK_CLASS_ORDER):
        out_path = os.path.join(SAVE_DIR, f"soft_prob_hist_{cname}.png")
        save_hist(
            soft[:, i],
            bins=HIST_BINS,
            title=f"Soft probability distribution: {cname}",
            xlabel=f"P({cname})",
            out_path=out_path,
        )

    # (2) max_conf distribution
    out_path = os.path.join(SAVE_DIR, "soft_max_conf_hist.png")
    save_hist(
        max_conf,
        bins=HIST_BINS,
        title="Soft max confidence (max_conf) distribution",
        xlabel="max_conf",
        out_path=out_path,
    )

    # (3) margin distribution
    out_path = os.path.join(SAVE_DIR, "soft_margin_hist.png")
    save_hist(
        margin,
        bins=HIST_BINS,
        title="Soft certainty margin distribution (top1 - top2)",
        xlabel="margin",
        out_path=out_path,
    )

    # (4) entropy distribution
    out_path = os.path.join(SAVE_DIR, "soft_entropy_hist.png")
    save_hist(
        entropy,
        bins=HIST_BINS,
        title="Soft entropy distribution (higher = more uncertain)",
        xlabel="entropy",
        out_path=out_path,
    )

    # (5) argmax class ratio
    out_path = os.path.join(SAVE_DIR, "soft_argmax_class_ratio.png")
    save_bar(
        BLACK_CLASS_ORDER,
        ratios,
        title="Soft argmax class ratio",
        xlabel="Class",
        ylabel="Ratio",
        out_path=out_path,
    )

    # (6) class-conditional max_conf distributions (grouped by argmax)
    for i, cname in enumerate(BLACK_CLASS_ORDER):
        mask = (argmax == i)
        if mask.sum() == 0:
            continue
        out_path = os.path.join(SAVE_DIR, f"soft_max_conf_hist_given_pred_{cname}.png")
        save_hist(
            max_conf[mask],
            bins=HIST_BINS,
            title=f"max_conf distribution | argmax = {cname}",
            xlabel="max_conf",
            out_path=out_path,
        )

    print("\n[DONE] Stats + plots saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()
