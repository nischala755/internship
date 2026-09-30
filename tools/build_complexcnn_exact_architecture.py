from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parents[1] / "ppt_assets_verified" / "guide_slide04_complexcnn_exact_architecture.png"


def box(ax, x, y, w, h, title, body, color, title_size=9.2, body_size=7.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25, edgecolor="#233746", facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center",
            fontsize=title_size, weight="bold", color="#14232d")
    ax.text(x + w / 2, y + h * 0.37, body, ha="center", va="center",
            fontsize=body_size, color="#243640", linespacing=1.22)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
        linewidth=1.25, color="#344b5a", shrinkA=2, shrinkB=2,
    ))


def main():
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.965, "Implemented ComplexCNN classifier: exact notebook architecture",
            ha="center", va="top", fontsize=20, weight="bold", color="#172a36")
    ax.text(0.5, 0.925,
            "Four complex polarimetric channels | classification only | 2,683,942 trainable parameters",
            ha="center", va="top", fontsize=11, color="#526774")

    box(ax, 0.015, 0.66, 0.105, 0.16, "Complex input",
        "[B, 4, 256, 256]\nHH, HV, VH, VV\ncomplex64", "#e6f0f7")
    box(ax, 0.14, 0.66, 0.13, 0.16, "Train-only standardization",
        "per channel, separately:\n(real - mu_r) / sigma_r\n(imag - mu_i) / sigma_i", "#edf3df")
    arrow(ax, 0.12, 0.74, 0.14, 0.74)

    ax.text(0.30, 0.875, "COMPLEX FEATURE PATH", fontsize=10.5, weight="bold", color="#175a78")
    top_y, h = 0.70, 0.14
    top = [
        (0.30, 0.115, "Stem", "CConv 5x5/s2: 4->32\nCBN + modReLU\n[32, 128, 128]"),
        (0.43, 0.125, "Stage 1: 2 ResBlocks", "32->48, s2 + 1x1 proj\n48->48, s1\n[48, 64, 64]"),
        (0.57, 0.125, "Stage 2: 2 ResBlocks", "48->96, s2 + 1x1 proj\n96->96, s1\n[96, 32, 32]"),
        (0.71, 0.125, "Stage 3: 2 ResBlocks", "96->160, s2 + 1x1 proj\n160->160, s1\n[160, 16, 16]"),
        (0.85, 0.135, "Complex FC feature", "concat real+imag -> 320\nGAP -> FC 320->192->128\nGELU, then Tanh"),
    ]
    for x, w, title, body in top:
        box(ax, x, top_y, w, h, title, body, "#dcecf5", title_size=8.6, body_size=7.1)
    arrow(ax, 0.27, 0.74, 0.30, 0.77)
    for (x, w, *_), (nx, *_rest) in zip(top, top[1:]):
        arrow(ax, x + w, top_y + h / 2, nx, top_y + h / 2)

    ax.text(0.30, 0.565, "EXPLICIT MAGNITUDE-PHASE PATH", fontsize=10.5, weight="bold", color="#7a4c12")
    bottom_y = 0.39
    bottom = [
        (0.30, 0.13, "Feature construction", "abs(x): 4 channels\nangle(x)/pi: 4 channels\nconcat -> [8, 256, 256]"),
        (0.45, 0.13, "Real Conv block 1", "Conv 5x5/s2: 8->32\nBN + GELU\n[32, 128, 128]"),
        (0.60, 0.12, "Real Conv block 2", "Conv 3x3/s2: 32->64\nBN + GELU\n[64, 64, 64]"),
        (0.74, 0.12, "Real Conv block 3", "Conv 3x3/s2: 64->128\nBN + GELU\n[128, 32, 32]"),
        (0.88, 0.105, "Magnitude-phase FC", "adaptive GAP -> 128\nFC 128->128\nTanh"),
    ]
    for x, w, title, body in bottom:
        box(ax, x, bottom_y, w, h, title, body, "#f5ead7", title_size=8.5, body_size=7.0)
    arrow(ax, 0.27, 0.70, 0.30, bottom_y + h / 2)
    for (x, w, *_), (nx, *_rest) in zip(bottom, bottom[1:]):
        arrow(ax, x + w, bottom_y + h / 2, nx, bottom_y + h / 2)

    box(ax, 0.52, 0.18, 0.16, 0.12, "Feature fusion",
        "128 complex + 128 mag-phase\nconcatenate -> 256", "#eee5cc")
    box(ax, 0.73, 0.18, 0.19, 0.12, "Six-class classifier",
        "FC 256->128 + GELU\nDropout 0.20 + FC 128->6", "#f1dddd")
    arrow(ax, 0.917, top_y, 0.65, 0.30)
    arrow(ax, 0.932, bottom_y, 0.65, 0.30)
    arrow(ax, 0.68, 0.24, 0.73, 0.24)

    ax.text(0.02, 0.27, "Exact module count", fontsize=10, weight="bold", color="#172a36")
    ax.text(0.02, 0.235,
            "6 complex residual blocks (2 per stage)\n"
            "16 ComplexConv2d modules = 32 real Conv2d submodules\n"
            "16 ComplexBatchNorm2d modules = 32 real BatchNorm2d modules\n"
            "13 complex modReLU activations; 3 projection shortcuts\n"
            "Magnitude-phase path: 3 real Conv2d blocks\n"
            "Five Linear layers overall",
            fontsize=8.5, va="top", color="#314650", linespacing=1.28)

    ax.text(0.02, 0.095,
            "Leakage control: mu and sigma are fitted on the 1,314 training images only and reused unchanged for the 162 validation and 162 test images. "
            "For each K-fold run, statistics are recomputed from that fold's training subset only.",
            fontsize=9.1, color="#355f3b", weight="bold", va="top")
    ax.text(0.02, 0.048,
            "Critical note: abs(x) and angle(x) are calculated after real/imaginary standardization. These are normalized-domain features; they should not be described as untouched physical magnitude and phase.",
            fontsize=8.8, color="#8a3e2f", va="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
