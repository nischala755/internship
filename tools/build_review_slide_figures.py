"""Create two missing, source-grounded visuals for the ten-slide review."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1] / "ppt_assets_verified"
CURVES = ROOT / "recovered_curves"


def box(axis, x, y, w, h, title, detail, color):
    axis.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#304152", linewidth=1.4))
    title_size = 9.5 if w < 2 else 11
    detail_size = 8.5 if w < 2 else 9.5
    axis.text(x + w / 2, y + h * 0.67, title, ha="center", va="center",
              fontsize=title_size, weight="bold")
    axis.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center",
              fontsize=detail_size, linespacing=1.1)


def arrow(axis, start, end):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                   linewidth=1.5, color="#304152"))


def clean_complexcnn_architecture():
    fig, ax = plt.subplots(figsize=(13.2, 5.1))
    ax.set(xlim=(0, 13.2), ylim=(0, 5.1))
    ax.axis("off")
    ax.text(6.6, 4.85, "Clean ComplexCNN classifier: actual notebook architecture",
            ha="center", fontsize=17, weight="bold")
    box(ax, .25, 2.05, 1.5, 1.05, "Complex input", "HH / HV / VH / VV", "#dfecf3")
    box(ax, 2.2, 3.10, 2.5, 1.00, "Complex CNN", "Stem 32; stages\n48 / 96 / 160", "#d9e8f4")
    box(ax, 5.1, 3.10, 2.05, 1.00, "Real + imag", "Pool 320 -> FC 128", "#d9e8f4")
    box(ax, 2.2, .95, 2.5, 1.00, "Magnitude + phase", "|x|, angle(x) / pi", "#e7efe4")
    box(ax, 5.1, .95, 2.05, 1.00, "Real CNN branch", "8 channels -> FC 128", "#e7efe4")
    box(ax, 7.65, 2.05, 2.15, 1.05, "Concatenate", "128 + 128 = 256", "#f4e9d5")
    box(ax, 10.25, 2.05, 2.55, 1.05, "Classifier", "FC 128 -> 6 classes", "#f2e0df")
    arrow(ax, (1.75, 2.82), (2.2, 3.5))
    arrow(ax, (1.75, 2.32), (2.2, 1.55))
    arrow(ax, (4.7, 3.6), (5.1, 3.6))
    arrow(ax, (4.7, 1.45), (5.1, 1.45))
    arrow(ax, (7.15, 3.6), (7.65, 2.86))
    arrow(ax, (7.15, 1.45), (7.65, 2.3))
    arrow(ax, (9.8, 2.57), (10.25, 2.57))
    ax.text(6.6, .28,
            "Input standardization is fitted on training data only. Classification head only; no detector/objectness head.",
            ha="center", fontsize=10.5, color="#435361")
    fig.savefig(ROOT / "guide_slide04_clean_complexcnn_architecture.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def read_rows(name):
    with (CURVES / name).open(newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def detection_capacity_panel():
    small = read_rows("yolo11s-vv_epochs.csv")
    medium = read_rows("yolo11m-vv_epochs.csv")
    fig = plt.figure(figsize=(13, 7.7))
    grid = fig.add_gridspec(2, 6, height_ratios=(1, 1.12), hspace=.39, wspace=.5)
    train_ax = fig.add_subplot(grid[0, :3])
    val_ax = fig.add_subplot(grid[0, 3:])
    axes = [fig.add_subplot(grid[1, start:start + 2]) for start in (0, 2, 4)]
    for rows, label, color in ((small, "YOLO11s", "#2b6cb0"),
                               (medium, "YOLO11m", "#c05621")):
        train_ax.plot([r["epoch"] for r in rows], [r["train_box_loss"] for r in rows],
                      label=label, color=color, linewidth=1.8)
        for ax, key in zip(axes, ("valid_precision", "valid_recall", "valid_f1_harmonic")):
            ax.plot([r["epoch"] for r in rows], [r[key] for r in rows],
                    label=label, color=color, linewidth=1.6)
    train_ax.set(title="Training box loss (logged)", xlabel="Epoch", ylabel="Loss")
    train_ax.legend(frameon=False)
    original = Image.open(ROOT / "yolo11s_yolo11m_validation_loss_and_map.png")
    validation_box_panel = original.crop((0, 0, original.width // 2, original.height))
    val_ax.imshow(validation_box_panel)
    val_ax.axis("off")
    val_ax.set_title("Validation box loss (original notebook plot)", fontsize=12)
    for ax, title in zip(axes, ("Validation precision", "Validation recall", "Validation harmonic F1")):
        ax.set(title=title, xlabel="Epoch", ylabel="Score", ylim=(0, 1))
        ax.legend(frameon=False)
    for ax in (train_ax, *axes):
        ax.grid(alpha=.2)
    fig.suptitle("YOLO11s vs YOLO11m | VV input | 100 epochs", fontsize=17, weight="bold")
    fig.savefig(ROOT / "guide_slide07_yolo11s_m_capacity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def completed_and_pending_evaluation():
    rows = [
        ("COMPLETED", "ComplexCNN classification", "5 folds; held-out accuracy 97.53%"),
        ("COMPLETED", "YOLO11n: VV / Pauli / learned", "50 epochs; test mAP50 82.1 / 78.3 / 74.3%"),
        ("COMPLETED", "YOLO11s / YOLO11m: VV", "100 epochs; test mAP50 85.28 / 87.05%"),
        ("COMPLETED", "YOLO26s / YOLO26m: VV", "100 epochs; test mAP50 85.90 / 88.02%"),
        ("COMPLETED", "Non-YOLO OBB prototypes", "CenterNet-OBB and DETR-OBB-Lite runs"),
        ("NEXT", "Matched detector comparison", "Same image IDs, preprocessing and thresholds"),
        ("NEXT", "View-generalization tests", "Leave-one-elevation-out; grouped azimuth"),
        ("NEXT", "Robustness and uncertainty", "Noise/SNR stress; repeated seeds; class-wise AP"),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5.1))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=("Status", "Experiment / protocol", "Evidence or next check"),
                     colWidths=(.14, .34, .52), cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c5d2d8")
        if row == 0:
            cell.set_facecolor("#294e60")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            completed = rows[row - 1][0] == "COMPLETED"
            cell.set_facecolor("#e8f2eb" if completed else "#fcf2e4")
            if col == 0:
                cell.get_text().set_color("#246647" if completed else "#a75d18")
                cell.get_text().set_weight("bold")
    ax.set_title("Completed evidence and remaining evaluation checks", pad=20,
                 fontsize=16, weight="bold")
    fig.text(.5, .035,
             "Reported test results come from separate runs; matched evaluation is next. "
             "YOLO26 results: supplied Colab screenshots.",
             ha="center", fontsize=9, color="#87503b")
    fig.savefig(ROOT / "guide_slide09_completed_vs_next_evaluation.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def revised_timeline():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set(xlim=(0, 4), ylim=(0, 1))
    ax.axis("off")
    ax.plot([.25, 3.75], [.49, .49], color="#6f8993", linewidth=2)
    milestones = (
        ("20-30 Sep", "Consolidate runs\nand evaluation protocol"),
        ("October", "Augmentation +\npaper baselines"),
        ("November", "Phase-aware fusion +\ncross-view tests"),
        ("December", "Ablations, repeated runs\nand manuscript"),
    )
    for index, (month, work) in enumerate(milestones):
        x = .48 + index * 1.02
        ax.scatter([x], [.49], s=170, color="#2d6e79", zorder=3)
        ax.text(x, .70, month, ha="center", weight="bold", fontsize=12)
        ax.text(x, .28, work, ha="center", va="center", fontsize=10)
    ax.text(3.85, .88, "End: 31 Dec 2026", ha="right", fontsize=11,
            weight="bold", color="#a4493f")
    fig.savefig(ROOT / "guide_slide10_revised_project_timeline.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    clean_complexcnn_architecture()
    detection_capacity_panel()
    completed_and_pending_evaluation()
    revised_timeline()
    print("Wrote slide 4, 7, 9 and 10 review figures to", ROOT)
