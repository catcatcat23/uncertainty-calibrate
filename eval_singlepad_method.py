# eval_singlepad_methods.py
"""
Evaluate MC Dropout / Deep Ensemble / Laplace on singlepad:
- Build hard labels from soft labels (argmax over NONE/INSUFFICIENT/PSEUDO).
- For each method, compute:
    * overall accuracy
    * per-class recall
    * (NEW) accuracy/recall on high-confidence subset (max prob >= 0.5 / 0.8)
- Additionally, treat 'defect' (undersolder or pseudosolder) as positive class
  and plot F1 score vs confidence threshold for the three methods on one figure.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== 路径配置 =====================
CSV_PATH = (
    "/home/cat/workspace/defect_data/"
    "defect_DA758_black_uuid_250310/send2terminal/250310/checked_samples.csv"
)
SAVE_DIR = os.path.dirname(CSV_PATH)

PART_NAME = "singlepad"   # 如果有多种 part，就只评估 singlepad
SPLIT = None              # 想只看 test 就改成 "test"，否则用全部样本

CLASS_ORDER = ["ok", "undersolder", "pseudosolder"]
N_CLASSES = 3

# F1 曲线的阈值个数
N_THRESH = 101  # 0, 0.01, ..., 1.0

# 新增：置信度阈值列表（对 3 类预测来说，max prob >= 该值才算“可信”）
CONF_THRESH_LIST = [0.5, 0.8]


# ===================== 工具函数 =====================

def confusion_matrix_3cls(y_true: np.ndarray, y_pred: np.ndarray):
    """手写一个 3x3 混淆矩阵，行是真值，列是预测。"""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def calc_precision_recall_f1_binary(y_true_bin: np.ndarray,
                                    scores: np.ndarray,
                                    thresholds: np.ndarray):
    """
    y_true_bin: [N], 0/1 (1 = positive, here 'defect')
    scores:     [N], 模型给出的正类分数，如 P(defect)
    thresholds: [T], 一组阈值

    返回:
      precisions, recalls, f1s, thresholds
    """
    y_true_bin = y_true_bin.astype(int)

    precisions = []
    recalls = []
    f1s = []

    for thr in thresholds:
        y_pred_bin = (scores >= thr).astype(int)

        tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
        fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
        fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    thresholds = np.asarray(thresholds, dtype=float)
    return (
        np.array(precisions, dtype=float),
        np.array(recalls, dtype=float),
        np.array(f1s, dtype=float),
        thresholds,
    )


def main():
    # ========== 读 CSV & 过滤 ==========
    df = pd.read_csv(CSV_PATH)

    if "part_name" in df.columns:
        df = df[df["part_name"] == PART_NAME]

    if SPLIT is not None and "split" in df.columns:
        df = df[df["split"] == SPLIT]

    df = df.reset_index(drop=True)
    print(f"使用样本数: {len(df)}")

    # ========== 构造真值硬标签（3 类） ==========
    soft_probs = df[[
        "NONE_CONF",
        "INSUFFICIENT_SOLDER_CONF",
        "PSEUDO_SOLDER_CONF",
    ]].to_numpy(dtype=np.float32)
    y_true = soft_probs.argmax(axis=1)  # [N], 0/1/2 对应 CLASS_ORDER

    # 另外构造“缺陷 vs ok”的二分类真值：
    # positive = defect = (undersolder/pseudosolder)
    y_true_defect = (y_true != 0).astype(int)

    # ========== 三种方法的列配置 ==========
    methods = {
        "MC Dropout": {
            "cols": [
                "NONE_CONF_DROPOUT",
                "INSUFFICIENT_SOLDER_CONF_DROPOUT",
                "PSEUDO_SOLDER_CONF_DROPOUT",
            ],
            "color": "tab:blue",
            "marker": "o",
        },
        "Deep Ensemble": {
            "cols": [
                "NONE_CONF_ENSEMBLE",
                "INSUFFICIENT_SOLDER_CONF_ENSEMBLE",
                "PSEUDO_SOLDER_CONF_ENSEMBLE",
            ],
            "color": "tab:orange",
            "marker": "s",
        },
        "Laplace": {
            "cols": [
                "NONE_CONF_LAPLACE",
                "INSUFFICIENT_SOLDER_CONF_LAPLACE",
                "PSEUDO_SOLDER_CONF_LAPLACE",
            ],
            "color": "tab:green",
            "marker": "D",
        },
    }

    # 为 F1 曲线记录各方法的数据
    thresholds = np.linspace(0.0, 1.0, N_THRESH)
    f1_curves = {}

    print("\n===== 分类性能（3 类） =====")
    for method_name, cfg in methods.items():
        prob_cols = cfg["cols"]
        probs = df[prob_cols].to_numpy(dtype=np.float32)  # [N, 3]

        # 3 类 argmax 预测
        y_pred = probs.argmax(axis=1)
        max_conf = probs.max(axis=1)  # 每个样本的最大类别置信度

        # overall accuracy（不加阈值）
        acc = (y_pred == y_true).mean()

        # confusion & per-class recall（不加阈值）
        cm = confusion_matrix_3cls(y_true, y_pred)
        recalls = []
        for c in range(N_CLASSES):
            support = cm[c].sum()
            rec = cm[c, c] / support if support > 0 else 0.0
            recalls.append(rec)

        print(f"\n[{method_name}]")
        print(f"  Overall accuracy (no conf filter): {acc:.4f}")
        for c, rec in enumerate(recalls):
            print(f"  Recall({CLASS_ORDER[c]}): {rec:.4f} "
                  f"(support={cm[c].sum()})")

        # ====== 新增：按 max prob 阈值过滤的 3 类性能 ======
        for conf_thr in CONF_THRESH_LIST:
            mask = max_conf >= conf_thr
            num_sel = int(mask.sum())
            coverage = num_sel / len(max_conf)

            if num_sel == 0:
                print(f"  [conf >= {conf_thr:.2f}] no samples selected.")
                continue

            y_true_sub = y_true[mask]
            y_pred_sub = y_pred[mask]

            acc_sub = (y_true_sub == y_pred_sub).mean()
            cm_sub = confusion_matrix_3cls(y_true_sub, y_pred_sub)

            recalls_sub = []
            for c in range(N_CLASSES):
                support_sub = cm_sub[c].sum()
                rec_sub = cm_sub[c, c] / support_sub if support_sub > 0 else 0.0
                recalls_sub.append(rec_sub)

            print(f"  [conf >= {conf_thr:.2f}] coverage={coverage:.3f} "
                  f"({num_sel}/{len(max_conf)} samples)")
            print(f"    accuracy: {acc_sub:.4f}")
            for c, rec_sub in enumerate(recalls_sub):
                print(f"    Recall({CLASS_ORDER[c]}): {rec_sub:.4f} "
                      f"(support={cm_sub[c].sum()})")

        # ====== F1 曲线（缺陷 vs ok）======
        # 正类分数: P(defect) = 1 - P(ok)
        scores_defect = 1.0 - probs[:, 0]

        _, _, f1s, _ = calc_precision_recall_f1_binary(
            y_true_defect, scores_defect, thresholds
        )
        f1_curves[method_name] = {
            "f1": f1s,
            "color": cfg["color"],
            "marker": cfg["marker"],
        }

    # ========== 画 F1 曲线 ==========
    plt.figure(figsize=(8, 5))

    for method_name, info in f1_curves.items():
        f1s = info["f1"]
        color = info["color"]

        # 找到最大 F1 的点，用于打印
        best_idx = int(f1s.argmax())
        best_thr = thresholds[best_idx]
        best_f1 = f1s[best_idx]

        print(f"\n[{method_name}] F1 曲线:")
        print(f"  Max F1 = {best_f1:.4f} at threshold = {best_thr:.3f}")

        plt.plot(
            thresholds,
            f1s,
            label=f"{method_name} (max F1={best_f1:.3f} @ {best_thr:.2f})",
            color=color,
        )

    plt.xlabel("Threshold on P(defect)")
    plt.ylabel("F1 score (defect vs ok)")
    plt.title("singlepad F1 curve vs threshold (MC Dropout / Ensemble / Laplace)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best")

    out_fig = os.path.join(SAVE_DIR, "singlepad_f1_curve_defect_vs_ok.png")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nF1 曲线图已保存到: {out_fig}")


if __name__ == "__main__":
    main()
