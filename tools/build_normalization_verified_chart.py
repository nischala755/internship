from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "ppt_assets_verified" / "guide_slide05_normalization_controlled_comparison.png"

# Executed notebook outputs:
# raw:  POLISAR_ComplexCNN_Classification_CLEAN (1).ipynb
# norm: POLISAR_ComplexCNN_Classification_CLEAN (2).ipynb
ACC_PER_IMAGE = np.array([0.9358108108, 0.9288135593, 0.9491525424, 0.7966101695, 0.8949152542])
ACC_NORM = np.array([0.9966216216, 0.9796610169, 0.9796610169, 0.9932203390, 0.9661016949])
F1_PER_IMAGE = np.array([0.9354567649, 0.9252582611, 0.9488378182, 0.7451758459, 0.8898908361])
F1_NORM = np.array([0.9966316219, 0.9795811182, 0.9793535500, 0.9932598039, 0.9655408259])


def main():
    folds = np.arange(1, 6)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("white")

    axes[0].plot(folds, 100 * ACC_PER_IMAGE, "o-", linewidth=2.4, markersize=7,
                 color="#a94f3d", label="Per-image max-amplitude")
    axes[0].plot(folds, 100 * ACC_NORM, "o-", linewidth=2.4, markersize=7,
                 color="#216b8f", label="Fold-train z-score")
    axes[0].axhline(100 * ACC_PER_IMAGE.mean(), linestyle="--", linewidth=1.3,
                    color="#a94f3d", alpha=0.55)
    axes[0].axhline(100 * ACC_NORM.mean(), linestyle="--", linewidth=1.3,
                    color="#216b8f", alpha=0.55)
    axes[0].set(title="Validation accuracy for each fold", xlabel="Cross-validation fold",
                ylabel="Accuracy (%)", xticks=folds, ylim=(74, 101))
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].grid(alpha=0.22)

    labels = ["Accuracy", "Macro-F1"]
    raw_means = 100 * np.array([ACC_PER_IMAGE.mean(), F1_PER_IMAGE.mean()])
    norm_means = 100 * np.array([ACC_NORM.mean(), F1_NORM.mean()])
    raw_std = 100 * np.array([ACC_PER_IMAGE.std(), F1_PER_IMAGE.std()])
    norm_std = 100 * np.array([ACC_NORM.std(), F1_NORM.std()])
    x = np.arange(2)
    width = 0.34
    axes[1].bar(x - width / 2, raw_means, width, yerr=raw_std, capsize=5,
                color="#c66a56", label="Per-image max-amplitude")
    axes[1].bar(x + width / 2, norm_means, width, yerr=norm_std, capsize=5,
                color="#3b82a6", label="Fold-train z-score")
    for xpos, means, stds in ((x - width / 2, raw_means, raw_std),
                              (x + width / 2, norm_means, norm_std)):
        for xi, mean, std in zip(xpos, means, stds):
            axes[1].text(xi, mean + std + 0.8, f"{mean:.2f} +/- {std:.2f}",
                         ha="center", va="bottom", fontsize=8.5)
    axes[1].set(title="Five-fold mean and population SD", ylabel="Score (%)",
                xticks=x, xticklabels=labels, ylim=(75, 103))
    axes[1].grid(axis="y", alpha=0.22)

    fig.suptitle("ComplexCNN normalization strategies: matched five-fold runs",
                 fontsize=16, weight="bold", color="#172a36")
    fig.text(0.5, 0.045,
             "Same architecture, folds (seed 42), LR 3e-4, batch size 8 and 8-epoch limit. "
             "Each point is the epoch with best validation macro-F1 within that fold.",
             ha="center", fontsize=9.2, color="#465b67")
    fig.text(0.5, 0.012,
             "Compared pipelines: per-image x_c/max|x_c| (including post-augmentation renormalization) versus fold-train real/imaginary z-score; official test set not used.",
             ha="center", fontsize=9.2, color="#35613d")
    fig.tight_layout(rect=(0, 0.09, 1, 0.91))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
