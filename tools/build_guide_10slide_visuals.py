"""Create scientific visuals for the guide's ten-slide technical review."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT.parent / "Pol_ISARSpaceTarget1.0"
OUT = PROJECT / "ppt_assets_verified"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.facecolor": "white"})


def finish(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sample_images() -> tuple[dict[str, np.ndarray], list[tuple[np.ndarray, str]]]:
    base = "Capella_el_40_az_0"
    image_dir = DATASET / "TIFFImages" / "AmplitudeImages"
    images = {pol: np.asarray(Image.open(image_dir / f"{base}_{pol}.tiff").convert("L"))
              for pol in ("HH", "HV", "VH", "VV")}
    labels = []
    label_path = DATASET / "ImageSets" / "labelTxt" / "train" / f"{base}.txt"
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 10:
            labels.append((np.array(parts[:8], dtype=float).reshape(4, 2), parts[8]))
    return images, labels


images, labels = sample_images()

# Slide 2: four measured polarizations plus their actual component annotations.
fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), constrained_layout=True)
for ax, (pol, img) in zip(axes[:4], images.items()):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(pol)
    ax.set_xticks([]); ax.set_yticks([])
axes[4].imshow(images["VV"], cmap="gray", vmin=0, vmax=255)
colors = {"parabolic_antenna": "#00a88f", "solar_array": "#e66a2b",
          "polygon_body": "#517ed5", "plate_fitting": "#c74a62", "rod_fitting": "#a87abb"}
for coords, name in labels:
    axes[4].add_patch(Polygon(coords, fill=False, edgecolor=colors.get(name, "#f4ca55"), linewidth=1.2))
axes[4].set_title("VV + component OBBs")
axes[4].set_xticks([]); axes[4].set_yticks([])
fig.suptitle("One Pol-ISAR frame: Capella, elevation 40, azimuth 0", fontsize=13)
finish(fig, "guide_slide02_polarizations_and_real_obbs.png")


def draw_box(ax, x, y, w, h, label, color="#e9f0f3", edge="#345a6a", size=11):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.018",
                         linewidth=1.3, facecolor=color, edgecolor=edge)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size, color="#162c36")


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.5, color="#526974"))


# Slide 3: measured conventional-CNN baseline. This is a diagram, not a curve.
fig = plt.figure(figsize=(12, 4.2), constrained_layout=True)
ax_img = fig.add_axes((.03, .13, .21, .7))
ax_img.imshow(images["VV"], cmap="gray", vmin=0, vmax=255)
ax_img.set_title("VV magnitude (example)")
ax_img.set_xticks([]); ax_img.set_yticks([])
ax = fig.add_axes((.27, .08, .70, .81)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
draw_box(ax, .03, .42, .19, .19, "Four-channel\nmagnitude", size=11)
draw_box(ax, .32, .42, .19, .19, "ResNet18\n15 epochs", size=11)
draw_box(ax, .61, .42, .19, .19, "Six-class\noutput", size=11)
arrow(ax, (.22, .515), (.32, .515)); arrow(ax, (.51, .515), (.61, .515))
ax.text(.50, .19, "Test accuracy 41.36%   |   Macro-F1 38.05%", ha="center",
        va="center", fontsize=13, weight="bold", color="#b14d41")
ax.text(.50, .08, "Measured baseline; no train/validation curve is available in the exported notebook",
        ha="center", fontsize=9, color="#555")
finish(fig, "guide_slide03_resnet18_measured_baseline.png")

# Slide 8: proposed architecture. No performance number is shown.
fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(.5, .94, "PROPOSED: phase-aware polarimetric fusion for component OBB detection",
        ha="center", fontsize=14, weight="bold")
draw_box(ax, .015, .42, .18, .29, "Complex scattering\nHH  HV  VH  VV\n(256 x 256)", color="#eaf0f4")
draw_box(ax, .265, .42, .20, .29, "log magnitude\n+ relative phase\nsin/cos encoding", color="#e3f2ed")
draw_box(ax, .535, .42, .20, .29, "Small learnable\npolarimetric fusion\nmodule", color="#f2efe4")
draw_box(ax, .805, .42, .18, .29, "OBB detector\ncomponent class\n+ rotated box", color="#f1e7e9")
for a, b in [(.195, .265), (.465, .535), (.735, .805)]:
    arrow(ax, (a, .565), (b, .565))
ax.text(.5, .22, "Physics-informed encoding: relative phase is invariant to a common phase offset",
        ha="center", fontsize=11)
ax.text(.5, .12, "Cross-view consistency and polarimetric fusion remain hypotheses to test",
        ha="center", fontsize=10, color="#a34a3f")
finish(fig, "guide_slide08_proposed_physics_fusion_architecture.png")

# Slide 9: future evaluation matrix; status words prevent accidental result claims.
fig, ax = plt.subplots(figsize=(11, 4.7), constrained_layout=True)
ax.axis("off")
cols = ["Evaluation condition", "VV baseline", "Pauli fusion", "Learned fusion", "Paper method"]
rows = [
    ["Clean official split (162 test)", "Pending rerun", "Planned", "Planned", "Planned"],
    ["Leave-one-elevation-out", "Planned", "Planned", "Planned", "Planned"],
    ["Grouped azimuth holdout", "Planned", "Planned", "Planned", "Planned"],
    ["Noise / SNR stress", "Planned", "Planned", "Planned", "Planned"],
]
table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center",
                 colWidths=[.32, .16, .16, .18, .18])
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2.1)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#c4d0d4")
    cell.set_facecolor("#294e60" if row == 0 else ("#eef4f6" if row % 2 else "#ffffff"))
    if row == 0: cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
ax.set_title("Evaluation protocol and current execution status", pad=18, fontsize=14, weight="bold")
finish(fig, "guide_slide09_evaluation_matrix_status.png")

# Slide 10: milestones bounded by the actual project end date.
fig, ax = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
ax.set_xlim(0, 4); ax.set_ylim(0, 1); ax.axis("off")
ax.plot([.25, 3.75], [.49, .49], color="#6f8993", linewidth=2)
milestones = [
    ("20-30 Sep", "Duplicate audit\nand clean rerun"),
    ("October", "Augmentation +\npaper baselines"),
    ("November", "Fusion module +\ncross-view tests"),
    ("December", "Ablation, statistics\nand manuscript"),
]
for i, (month, work) in enumerate(milestones):
    x = .48 + i * 1.02
    ax.scatter([x], [.49], s=170, color="#2d6e79", zorder=3)
    ax.text(x, .70, month, ha="center", weight="bold", fontsize=12)
    ax.text(x, .28, work, ha="center", va="center", fontsize=10)
ax.text(3.85, .88, "End: 31 Dec 2026", ha="right", fontsize=11, weight="bold", color="#a4493f")
finish(fig, "guide_slide10_2026_project_timeline.png")

print("Created five slide visuals in", OUT)
