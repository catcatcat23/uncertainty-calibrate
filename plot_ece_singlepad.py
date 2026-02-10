# plot_classwise_pred_singlepad.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 基本路径配置 =========
CSV_PATH = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310/checked_samples.csv"
SAVE_DIR = os.path.dirname(CSV_PATH)

PART_NAME = "singlepad"   # 如果 CSV 里有别的 part，就只用 singlepad
SPLIT = None              # 想只用 test 就改成 "test"，否则用全部

CLASS_ORDER = ["ok", "undersolder", "pseudosolder"]
N_BINS = 10


# ========= ECE 计算：标准 reliability diagram 公式 =========
def compute_ece(confidences: np.ndarray,
                correct: np.ndarray,
                n_bins: int = 15):
    """
    标准 ECE:
      ECE = sum_k ( | acc_k - conf_k | * (#bin_k / N) )

    conf:   [N]，每个样本的 top-1 置信度
    correct:[N]，0/1 是否预测正确
    """
    assert confidences.shape == correct.shape
    N = len(confidences)
    if N == 0:
        # 没有样本的极端情况（某个类完全没被预测到）
        return 0.0, np.array([]), np.array([]), np.array([])

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    bin_centers = []
    bin_acc = []
    bin_conf = []

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        count_in_bin = in_bin.sum()
        if count_in_bin == 0:
            continue

        acc_in_bin = correct[in_bin].mean()
        conf_in_bin = confidences[in_bin].mean()
        prop_in_bin = count_in_bin / N

        ece += prop_in_bin * abs(acc_in_bin - conf_in_bin)

        bin_centers.append((bin_lower + bin_upper) / 2.0)
        bin_acc.append(acc_in_bin)
        bin_conf.append(conf_in_bin)

    return ece, np.array(bin_centers), np.array(bin_acc), np.array(bin_conf)


def main():
    # ========= 读 CSV & 构造真值标签 =========
    df = pd.read_csv(CSV_PATH)

    if "part_name" in df.columns:
        df = df[df["part_name"] == PART_NAME]

    if SPLIT is not None and "split" in df.columns:
        df = df[df["split"] == SPLIT]

    df = df.reset_index(drop=True)
    print(f"Num samples used: {len(df)}")

    # 软标签 -> 硬标签 y_true
    soft_probs = df[[
        "NONE_CONF",
        "INSUFFICIENT_SOLDER_CONF",
        "PSEUDO_SOLDER_CONF"
    ]].to_numpy(dtype=np.float32)
    y_true = soft_probs.argmax(axis=1)   # [N]

    # ========= 三个方法的概率列 =========
    methods = {
        "MC Dropout": {
            "cols": [
                "NONE_CONF_DROPOUT",
                "INSUFFICIENT_SOLDER_CONF_DROPOUT",
                "PSEUDO_SOLDER_CONF_DROPOUT",
            ],
            "color": "tab:blue",
            "marker": "o",
            "fname": "singlepad_classwise_pred_mcdropout.png",
        },
        "Deep Ensemble": {
            "cols": [
                "NONE_CONF_ENSEMBLE",
                "INSUFFICIENT_SOLDER_CONF_ENSEMBLE",
                "PSEUDO_SOLDER_CONF_ENSEMBLE",
            ],
            "color": "tab:orange",
            "marker": "s",
            "fname": "singlepad_classwise_pred_ensemble.png",
        },
        "Laplace": {
            "cols": [
                "NONE_CONF_LAPLACE",
                "INSUFFICIENT_SOLDER_CONF_LAPLACE",
                "PSEUDO_SOLDER_CONF_LAPLACE",
            ],
            "color": "tab:green",
            "marker": "D",
            "fname": "singlepad_classwise_pred_laplace.png",
        },
    }

    for method_name, cfg in methods.items():
        prob_cols = cfg["cols"]
        color = cfg["color"]
        marker = cfg["marker"]

        # [N, 3]
        probs = df[prob_cols].to_numpy(dtype=np.float32)
        y_pred = probs.argmax(axis=1)        # 预测类
        conf   = probs.max(axis=1)           # top-1 概率

        # 存每个 class 的 ECE，方便算 macro class-wise ECE
        class_eces = []

        fig, axes = plt.subplots(
            1, len(CLASS_ORDER),
            figsize=(15, 4),
            sharey=True
        )
        fig.suptitle(
            f"singlepad class-wise reliability (predicted class, {method_name})"
        )

        for c, (ax, class_name) in enumerate(zip(axes, CLASS_ORDER)):
            # 只看“预测为该类”的样本
            mask_c = (y_pred == c)
            conf_c = conf[mask_c]
            correct_c = (y_true[mask_c] == c).astype(np.float32)

            print(
                f"[{method_name}] class={class_name}, "
                f"num_pred={len(conf_c)}"
            )

            ece_c, centers, bin_acc, bin_conf = compute_ece(
                conf_c, correct_c, n_bins=N_BINS
            )
            class_eces.append(ece_c)

            # 画 diag 线
            ax.plot(
                [0, 1], [0, 1],
                linestyle="--",
                color="gray",
                linewidth=1.0,
                label="Perfect"
            )

            if len(centers) > 0:
                width = 1.0 / N_BINS

                # 柱子：bin 内经验准确率
                ax.bar(
                    centers,
                    bin_acc,
                    width=width,
                    alpha=0.3,
                    edgecolor=color,
                    color=color,
                    label="Accuracy"
                )

                # 折线：bin 内平均置信度
                ax.plot(
                    centers,
                    bin_conf,
                    marker=marker,
                    linewidth=2.0,
                    color=color,
                    label=f"Mean confidence (ECE={ece_c:.3f})"
                )
            else:
                # 理论上几乎不会出现，如果某个类从未被预测到
                ax.text(
                    0.5, 0.5,
                    "No predictions",
                    ha="center", va="center",
                    transform=ax.transAxes
                )

            ax.set_title(f"Class: {class_name}")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

            if c == 0:
                ax.set_ylabel("Empirical accuracy")
            ax.set_xlabel("Predicted confidence for predicted class")

            ax.legend(loc="lower right", fontsize=9)

        macro_ece = float(np.mean(class_eces))
        print(f"[{method_name}] class-wise macro ECE = {macro_ece:.4f}")

        plt.tight_layout()
        out_path = os.path.join(SAVE_DIR, cfg["fname"])
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
