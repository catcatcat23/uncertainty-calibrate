#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =========================
# CONFIG
# =========================
ROOT_DIR = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310"
CSV_PATH = os.path.join(ROOT_DIR, "calibrate_with_softlabels.csv")

# 输出目录区分
SAVE_DIR = os.path.join(os.path.dirname(CSV_PATH), "eval_temp_scaling_black")
os.makedirs(SAVE_DIR, exist_ok=True)

# 统一分箱
N_BINS_ECE = 15

# Black 3类配置
BLACK_CLASS_ORDER = ["ok", "INSUFFICIENT_SOLDER", "PSEUDO_SOLDER"]
BLACK_SOFT_COLS = [
    "soft_ok",
    "soft_INSUFFICIENT_SOLDER",
    "soft_PSEUDO_SOLDER",
]
OPTIONAL_SOFT_COLD = "soft_COLD_WELD"

plt.rcParams["font.family"] = "DejaVu Sans"

# =========================
# 基础数学/工具函数
# =========================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def softmax(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 Softmax"""
    e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def soft_to_logits(probs: np.ndarray, eps=1e-12) -> np.ndarray:
    """
    将概率逆转为 Logits (近似)。
    Logits = log(p)
    """
    probs = np.clip(probs, eps, 1.0)
    return np.log(probs)

def apply_temperature(logits: np.ndarray, temp) -> np.ndarray:
    """
    应用温度缩放。
    temp 可以是标量 (Global) 或 向量 (Class-wise)。
    """
    # 避免除以0或负数
    if np.ndim(temp) == 0:
        t = max(float(temp), 1e-3)
    else:
        t = np.maximum(np.array(temp), 1e-3).reshape(1, -1)
    
    scaled_logits = logits / t
    return softmax(scaled_logits)

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
    x = _normalize_rows(x)
    return x

# =========================
# ECE 计算核心
# =========================

def compute_ece_from_probs(probs: np.ndarray, y_true: np.ndarray, n_bins: int):
    """
    计算 Multi-class ECE
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == y_true).astype(float)
    
    return _calc_ece_bins(confidences, correctness, n_bins)

def compute_binary_ece_from_probs(probs: np.ndarray, y_true: np.ndarray, n_bins: int):
    """
    计算 Binary ECE (OK vs Defect)
    Class 0 is OK, Class 1,2... are Defect
    """
    # 二值化标签：0为OK，1为Defect
    y_true_bin = (y_true != 0).astype(int)
    
    # 二值化概率：Defect Prob = sum(defect_classes)
    prob_defect = probs[:, 1:].sum(axis=1)
    prob_ok = probs[:, 0]
    
    # 构造二分类的 confidence 和 prediction
    # 如果 prob_defect > 0.5 -> pred=1, conf=prob_defect
    # 否则 -> pred=0, conf=prob_ok
    predictions_bin = (prob_defect >= 0.5).astype(int)
    confidences_bin = np.where(predictions_bin == 1, prob_defect, prob_ok)
    
    correctness_bin = (predictions_bin == y_true_bin).astype(float)
    
    return _calc_ece_bins(confidences_bin, correctness_bin, n_bins)

def _calc_ece_bins(confidences, correctness, n_bins):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bin_edges[1:-1], right=True)
    
    ece = 0.0
    N = len(confidences)
    
    # 用于绘图的数据
    plot_data = {
        "bin_avg_conf": [],
        "bin_avg_acc": [],
        "bin_count": []
    }
    
    for b in range(n_bins):
        mask = (bin_ids == b)
        cnt = mask.sum()
        if cnt > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correctness[mask].mean()
            weight = cnt / N
            ece += weight * abs(avg_acc - avg_conf)
            
            plot_data["bin_avg_conf"].append(avg_conf)
            plot_data["bin_avg_acc"].append(avg_acc)
            plot_data["bin_count"].append(cnt)
        else:
            plot_data["bin_avg_conf"].append(0)
            plot_data["bin_avg_acc"].append(0)
            plot_data["bin_count"].append(0)
            
    return ece, plot_data

# =========================
# 优化器：寻找最佳温度
# =========================

def nll_loss(temp, logits, y_true):
    """
    Negative Log Likelihood Loss.
    优化目标通常是最小化 NLL，而不是直接最小化 ECE (NLL更平滑且凸)。
    """
    probs = apply_temperature(logits, temp)
    # 取出真实类别对应的概率
    true_probs = probs[np.arange(len(y_true)), y_true]
    # 避免 log(0)
    true_probs = np.clip(true_probs, 1e-12, 1.0)
    return -np.mean(np.log(true_probs))

def optimize_global_temp(logits, y_true):
    """全局温度缩放：寻找一个标量 T"""
    print("  > Optimizing Global Temperature (minimizing NLL)...")
    res = minimize(
        fun=nll_loss,
        x0=[1.5],
        args=(logits, y_true),
        bounds=[(0.01, 10.0)],
        method='L-BFGS-B'
    )
    best_t = res.x[0]
    print(f"  > Best Global T: {best_t:.4f}")
    return best_t

def optimize_classwise_temp(logits, y_true, n_classes):
    """按类温度缩放：寻找向量 T [c]"""
    print("  > Optimizing Class-wise Temperature...")
    
    # 初始猜测
    initial_guess = np.ones(n_classes) * 1.5
    
    res = minimize(
        fun=nll_loss,
        x0=initial_guess,
        args=(logits, y_true),
        bounds=[(0.01, 10.0)] * n_classes,
        method='L-BFGS-B'
    )
    best_t_vec = res.x
    print(f"  > Best Class-wise T: {best_t_vec}")
    return best_t_vec

# =========================
# 绘图
# =========================
def plot_reliability_diagram(plot_data, title, filename):
    """绘制可靠性图"""
    confs = np.array(plot_data["bin_avg_conf"])
    accs = np.array(plot_data["bin_avg_acc"])
    counts = np.array(plot_data["bin_count"])
    
    # 过滤掉空箱子以便绘图美观 (可选)
    mask = counts > 0
    confs = confs[mask]
    accs = accs[mask]
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.scatter(confs, accs, c='red', s=50, zorder=5)
    plt.bar(confs, accs, width=0.05, alpha=0.3, color='blue', edgecolor='black', label='Outputs')
    
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# =========================
# 主流程
# =========================

def main():
    ensure_dir(SAVE_DIR)
    
    # 1. 加载数据
    print(f"[Step 1] Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # 2. 预处理与过滤
    # 过滤 singlepad
    if "part_name" in df.columns:
        df = df[df["part_name"].str.lower() == "singlepad"].copy()
    
    # 过滤 black 版本
    if "version_folder" in df.columns:
        df = df[df["version_folder"].astype(str).str.lower().str.startswith("black")].copy()
    
    # 处理 Cold Weld (如果有则丢弃，同原逻辑)
    needed = BLACK_SOFT_COLS + [OPTIONAL_SOFT_COLD]
    if all(c in df.columns for c in needed):
        soft4 = ensure_probs_like(df[needed].to_numpy(float), "check_cold")
        y4 = soft4.argmax(axis=1)
        # Cold Weld is index 3
        df = df[y4 != 3].copy()
    
    print(f"  > Data Valid Rows: {len(df)}")
    if len(df) == 0:
        return

    # 3. 提取 Soft Labels 和 Label
    soft_probs = ensure_probs_like(df[BLACK_SOFT_COLS].to_numpy(float), "final_soft")
    # 假设 soft label 的 argmax 就是伪真值 (Self-Knowledge Distillation 场景) 
    # 或者 CSV 里有 gt_label。这里沿用原逻辑：y_true由soft argmax决定，
    # *但在校准场景下，通常需要真实的硬标签 (GT)。* # *如果您的 CSV 里有真实的 'label' 列，请取消下面注释并使用它。*
    # if "label" in df.columns:
    #     y_true = df["label"].values # 需要映射到 0,1,2
    # else:
    y_true = soft_probs.argmax(axis=1) 

    # 4. 转换为 Logits
    logits = soft_to_logits(soft_probs)
    
    results_summary = []

    # ==========================================
    # Phase A: Original (Unscaled, T=1)
    # ==========================================
    print("\n[Phase A] Evaluating Original (T=1.0)...")
    ece_multi, data_multi = compute_ece_from_probs(soft_probs, y_true, N_BINS_ECE)
    ece_bin, data_bin = compute_binary_ece_from_probs(soft_probs, y_true, N_BINS_ECE)
    
    plot_reliability_diagram(data_multi, f"Original Multi-Class (ECE={ece_multi:.4f})", 
                             os.path.join(SAVE_DIR, "rel_diag_original_multi.png"))
    plot_reliability_diagram(data_bin, f"Original Binary (ECE={ece_bin:.4f})", 
                             os.path.join(SAVE_DIR, "rel_diag_original_binary.png"))
    
    results_summary.append({
        "Method": "Original (No Scaling)",
        "Temp": "1.0",
        "Multi-Class ECE": ece_multi,
        "Binary ECE": ece_bin
    })

    # ==========================================
    # Phase B: Global Temperature Scaling
    # ==========================================
    print("\n[Phase B] Global Temperature Scaling...")
    # 1. 寻找最佳 T
    best_t_global = optimize_global_temp(logits, y_true)
    
    # 2. 应用最佳 T
    probs_global = apply_temperature(logits, best_t_global)
    
    # 3. 评估
    ece_multi_g, data_multi_g = compute_ece_from_probs(probs_global, y_true, N_BINS_ECE)
    ece_bin_g, data_bin_g = compute_binary_ece_from_probs(probs_global, y_true, N_BINS_ECE)
    
    plot_reliability_diagram(data_multi_g, f"Global TS (T={best_t_global:.2f}, ECE={ece_multi_g:.4f})", 
                             os.path.join(SAVE_DIR, "rel_diag_global_multi.png"))
    
    results_summary.append({
        "Method": "Global Temp Scaling",
        "Temp": f"{best_t_global:.4f}",
        "Multi-Class ECE": ece_multi_g,
        "Binary ECE": ece_bin_g
    })

    # ==========================================
    # Phase C: Class-wise Temperature Scaling
    # ==========================================
    print("\n[Phase C] Class-wise Temperature Scaling...")
    # 1. 寻找最佳 T 向量
    n_classes = len(BLACK_CLASS_ORDER)
    best_t_cw = optimize_classwise_temp(logits, y_true, n_classes)
    
    # 2. 应用最佳 T
    probs_cw = apply_temperature(logits, best_t_cw)
    
    # 3. 评估
    ece_multi_c, data_multi_c = compute_ece_from_probs(probs_cw, y_true, N_BINS_ECE)
    ece_bin_c, data_bin_c = compute_binary_ece_from_probs(probs_cw, y_true, N_BINS_ECE)

    plot_reliability_diagram(data_multi_c, f"Class-wise TS (ECE={ece_multi_c:.4f})", 
                             os.path.join(SAVE_DIR, "rel_diag_classwise_multi.png"))

    results_summary.append({
        "Method": "Class-wise Temp Scaling",
        "Temp": str(np.round(best_t_cw, 3)),
        "Multi-Class ECE": ece_multi_c,
        "Binary ECE": ece_bin_c
    })

    # =========================
    # 汇总输出
    # =========================
    df_res = pd.DataFrame(results_summary)
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(df_res.to_markdown(index=False))
    
    csv_out = os.path.join(SAVE_DIR, "calibration_summary.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\nResults saved to {csv_out}")
    print(f"Reliability diagrams saved in {SAVE_DIR}")

if __name__ == "__main__":
    main()