"""Plot reported 100-epoch YOLO OBB results with run provenance kept explicit."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# YOLO26 values were transcribed from the researcher's executed Colab screenshot.
# YOLO11 values are from the executed larger-model notebook and its saved summary.
RUNS = {
    "YOLO11s": {"map50": 0.852763, "map95": 0.628999, "source": "YOLO11 notebook"},
    "YOLO11m": {"map50": 0.870462, "map95": 0.652411, "source": "YOLO11 notebook"},
    "YOLO26s": {"map50": 0.859038, "map95": 0.650149, "source": "user Colab screenshot"},
    "YOLO26m": {"map50": 0.880184, "map95": 0.666971, "source": "user Colab screenshot"},
}


def main():
    out = Path(__file__).resolve().parents[1] / "ppt_assets_verified"
    names = list(RUNS)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    width = 0.34
    for offset, key, label, color in ((-width / 2, "map50", "Test mAP50", "#2b6cb0"),
                                       (width / 2, "map95", "Test mAP50-95", "#c05621")):
        bars = ax.bar(x + offset, [100 * RUNS[name][key] for name in names],
                      width=width, color=color, label=label)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .6,
                    f"{bar.get_height():.1f}", ha="center", fontsize=9)
    ax.axvline(1.5, color="#84929c", linewidth=1)
    ax.set_xticks(x, ["YOLO11s\nVV", "YOLO11m\nVV", "YOLO26s\nVV", "YOLO26m\nVV"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test mAP (%)")
    ax.set_title("100-epoch OBB runs: reported VV-only results", weight="bold")
    ax.text(.5, -.20, "YOLO11: executed notebook     |     YOLO26: supplied Colab screenshot",
            transform=ax.transAxes, ha="center", fontsize=9, color="#435361")
    ax.text(.5, -.28, "Separate runs; protocol-matched evaluation remains to be done",
            transform=ax.transAxes, ha="center", fontsize=9, color="#a04135")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.subplots_adjust(bottom=.28)
    path = out / "guide_slide07_100epoch_yolo11_yolo26_reported_results.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
