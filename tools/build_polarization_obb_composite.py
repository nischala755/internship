"""Build slide-ready polarimetric examples with all source OBB annotations."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT.parent / "Pol_ISARSpaceTarget1.0"
IMAGE_DIR = DATASET / "TIFFImages" / "AmplitudeImages"
LABEL_DIR = DATASET / "ImageSets" / "labelTxt" / "train"
OUT = PROJECT / "ppt_assets_verified"

SAMPLES = (
    ("Capella_el_40_az_0", "Capella\nelevation 40 deg | azimuth 0 deg"),
    ("HISEA1_el_45_az_39", "HISEA1\nelevation 45 deg | azimuth 39 deg"),
    ("Worldview4_el_40_az_0", "Worldview4\nelevation 40 deg | azimuth 0 deg"),
)
POLARIZATIONS = ("HH", "HV", "VH", "VV")
CLASS_COLORS = {
    "cuboid_body": "#ffd166",
    "plate_body": "#8dd3c7",
    "polygon_body": "#4f7cff",
    "optical_lens": "#ff66c4",
    "panel_antenna": "#7bd389",
    "parabolic_antenna": "#00b8a9",
    "solar_array": "#ff8c42",
    "plate_fitting": "#ef476f",
    "rod_fitting": "#9b5de5",
}


def read_dota_annotations(path: Path):
    annotations = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 10:
            raise ValueError(f"Malformed DOTA annotation at {path}:{line_number}")
        annotations.append({
            "points": np.asarray(parts[:8], dtype=float).reshape(4, 2),
            "class_name": parts[8],
            "difficult": int(parts[9]),
        })
    return annotations


def load_amplitude(base_name: str, polarization: str):
    path = IMAGE_DIR / f"{base_name}_{polarization}.tiff"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("L"))


def class_summary(annotations):
    counts = Counter(item["class_name"] for item in annotations)
    return ", ".join(f"{name.replace('_', ' ')} x{counts[name]}" for name in sorted(counts))


def draw_composite():
    fig, axes = plt.subplots(3, 5, figsize=(16, 8.7), constrained_layout=False)
    fig.patch.set_facecolor("white")
    column_titles = ("HH", "HV", "VH", "VV", "VV + all component OBBs")
    visible_classes = set()

    for row, (base_name, row_title) in enumerate(SAMPLES):
        annotations = read_dota_annotations(LABEL_DIR / f"{base_name}.txt")
        visible_classes.update(item["class_name"] for item in annotations)
        images = {pol: load_amplitude(base_name, pol) for pol in POLARIZATIONS}
        for column, polarization in enumerate(POLARIZATIONS):
            axis = axes[row, column]
            axis.imshow(images[polarization], cmap="gray", vmin=0, vmax=255)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(column_titles[column], fontsize=12, weight="bold", pad=5)

        obb_axis = axes[row, 4]
        obb_axis.imshow(images["VV"], cmap="gray", vmin=0, vmax=255)
        for item in annotations:
            obb_axis.add_patch(Polygon(item["points"], closed=True, fill=False,
                                       edgecolor=CLASS_COLORS[item["class_name"]],
                                       linewidth=1.65))
        obb_axis.set_xticks([])
        obb_axis.set_yticks([])
        if row == 0:
            obb_axis.set_title(column_titles[4], fontsize=12, weight="bold", pad=5)

        axes[row, 0].text(.025, .965, row_title, transform=axes[row, 0].transAxes,
                          ha="left", va="top", fontsize=8.7, weight="bold", color="white",
                          bbox={"facecolor": "#18232b", "edgecolor": "none", "alpha": .78,
                                "pad": 3})
        obb_axis.text(.5, -.075, f"{len(annotations)} boxes: {class_summary(annotations)}",
                      transform=obb_axis.transAxes, ha="center", va="top", fontsize=7.3)

    handles = [Line2D([0], [0], color=CLASS_COLORS[name], linewidth=3,
                      label=name.replace("_", " ")) for name in CLASS_COLORS if name in visible_classes]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .018), fontsize=9.3, handlelength=2.2)
    fig.suptitle("Polarimetric ISAR appearance and component-level oriented bounding boxes",
                 fontsize=17, weight="bold", y=.985)
    fig.text(.5, .945,
             "Each row is one physical frame. HH/HV/VH/VV are polarization channels; every coloured polygon is read from its DOTA label file.",
             ha="center", fontsize=10.2, color="#425563")
    plt.subplots_adjust(left=.072, right=.992, top=.90, bottom=.13, wspace=.045, hspace=.22)
    path = OUT / "guide_slide_dataset_three_frames_all_obbs.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def draw_explanation():
    fig, ax = plt.subplots(figsize=(15, 3.8))
    ax.axis("off")
    ax.text(.01, .91, "How to read the polarimetric and annotation panels", fontsize=17,
            weight="bold", transform=ax.transAxes)
    channel_text = (
        "HH: horizontal transmit, horizontal receive\n"
        "HV: horizontal transmit, vertical receive\n"
        "VH: vertical transmit, horizontal receive\n"
        "VV: vertical transmit, vertical receive"
    )
    ax.text(.02, .70, channel_text, fontsize=11.2, va="top", linespacing=1.55,
            transform=ax.transAxes)
    ax.text(.35, .70,
            "OBB meaning\nEach outline is one labelled component.\n"
            "Four corners encode component orientation.\n"
            "Colour indicates the component class.\n"
            "All boxes come directly from the source labels.",
            fontsize=10.8, va="top", linespacing=1.45, transform=ax.transAxes)
    present = ("parabolic_antenna", "solar_array", "polygon_body", "plate_fitting",
               "rod_fitting", "cuboid_body", "panel_antenna", "optical_lens")
    for index, name in enumerate(present):
        x = .70 + (index % 2) * .16
        y = .70 - (index // 2) * .14
        ax.plot([x, x + .035], [y, y], color=CLASS_COLORS[name], linewidth=5,
                transform=ax.transAxes, solid_capstyle="butt")
        ax.text(x + .04, y, name.replace("_", " "), va="center", fontsize=8.8,
                transform=ax.transAxes)
    ax.text(.70, .08, "Not displayed in the selected examples: plate body",
            fontsize=9.5, color="#596873", transform=ax.transAxes)
    path = OUT / "guide_slide_dataset_obb_legend_explanation.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(draw_composite())
    print(draw_explanation())
