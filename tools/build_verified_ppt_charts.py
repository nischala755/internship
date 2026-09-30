"""Build PPT-ready charts only from verified executed-notebook results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "ppt_assets_verified"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})


def finish(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close()


folds = np.arange(1, 6)
acc_per_image = np.array([.935811, .928814, .949153, .796610, .894915])
acc_norm = np.array([.996622, .979661, .979661, .993220, .966102])
plt.figure(figsize=(8, 4.5))
plt.plot(folds, 100 * acc_per_image, "o-", linewidth=2, label="Per-image max-amplitude")
plt.plot(folds, 100 * acc_norm, "o-", linewidth=2, label="Fold-train statistics")
plt.axhline(100 * acc_per_image.mean(), linestyle="--", alpha=.5)
plt.axhline(100 * acc_norm.mean(), linestyle="--", alpha=.5)
plt.xticks(folds)
plt.ylim(75, 101)
plt.xlabel("Cross-validation fold")
plt.ylabel("Validation accuracy (%)")
plt.title("ComplexCNN: Five-Fold Normalization-Strategy Comparison")
plt.grid(alpha=.25)
plt.legend()
finish("classification_kfold_normalization_comparison.png")

cm = np.array([[23, 0, 0, 0, 3, 1], [0, 27, 0, 0, 0, 0], [0, 0, 27, 0, 0, 0],
               [0, 0, 0, 27, 0, 0], [0, 0, 0, 0, 27, 0], [0, 0, 0, 0, 0, 27]])
classes = ["Capella", "HISEA1", "QPSSAR", "Radarsat2", "StarLink", "Worldview4"]
plt.figure(figsize=(7, 5.8))
plt.imshow(cm, cmap="Blues")
plt.colorbar(label="Samples")
plt.xticks(range(6), classes, rotation=35, ha="right")
plt.yticks(range(6), classes)
for i in range(6):
    for j in range(6):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > 13 else "black")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("ComplexCNN Held-Out Test Confusion Matrix")
finish("classification_test_confusion_matrix.png")

models = ["YOLO11n", "YOLO11s", "YOLO11m"]
precision = [.804, .881305, .892630]
recall = [.819, .825588, .839673]
map50 = [.821, .852763, .870462]
map95 = [.582, .628999, .652411]
x = np.arange(len(models)); width = .2
plt.figure(figsize=(9, 4.8))
for offset, values, label in [(-1.5, precision, "Precision"), (-.5, recall, "Recall"),
                              (.5, map50, "mAP50"), (1.5, map95, "mAP50-95")]:
    plt.bar(x + offset * width, values, width, label=label)
plt.xticks(x, models)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("OBB Detector Capacity Comparison (Provisional 174-Image Test)")
plt.grid(axis="y", alpha=.25)
plt.legend(ncol=4, loc="lower center")
finish("yolo_capacity_metrics_provisional.png")

names = ["O-RTMDet", "O-FCOS", "O-FRCNN", "O-ATSS", "PSC", "S2A-Net"]
paper_map = [.6828, .6657, .6597, .6586, .6188, .4908]
plt.figure(figsize=(8.5, 4.5))
bars = plt.bar(names, 100 * np.array(paper_map), color=["#24557a", "#5b8f68", "#8a6d3b", "#7676a7", "#a85b5b", "#777777"])
for bar, value in zip(bars, paper_map):
    plt.text(bar.get_x() + bar.get_width()/2, 100*value + .7, f"{100*value:.1f}", ha="center", fontsize=9)
plt.ylim(0, 75)
plt.ylabel("Mean mAP50 across 12 view conditions (%)")
plt.title("Translated Paper: Cross-View OBB Method Comparison")
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=.25)
finish("translated_paper_method_map50.png")

tf_names = ["Random Forest\nmag+phase", "ResNet18\nmagnitude", "Complex CNN\nreal+imag", "ANOVA-F\n512 features"]
tf_acc = [1.0, .969697, .949495, .787879]
plt.figure(figsize=(8.5, 4.5))
bars = plt.bar(tf_names, 100 * np.array(tf_acc), color=["#406e8e", "#5b8f68", "#a85b5b", "#8a6d3b"])
for bar, value in zip(bars, tf_acc):
    plt.text(bar.get_x() + bar.get_width()/2, 100*value + .8, f"{100*value:.1f}", ha="center")
plt.ylim(0, 106)
plt.ylabel("Held-out accuracy (%)")
plt.title("TFPSSTD View-Disjoint Classification")
plt.grid(axis="y", alpha=.25)
finish("tfpsstd_classification_comparison.png")

print(f"Wrote verified charts to {OUT}")
