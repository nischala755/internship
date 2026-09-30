"""Plot exact Ultralytics OBB train/validation losses from saved results.csv files."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = {
    "YOLO11n-VV": "polisar_obb_outputs/runs_obb/baseline_single_vv_yolo_obb/results.csv",
    "YOLO11n-Pauli": "polisar_obb_outputs/runs_obb/proposed_pauli_fusion_yolo_obb/results.csv",
    "YOLO11n-LearnedFusion": "polisar_obb_outputs/runs_obb/proposed_learned_fusion_yolo_obb/results.csv",
    "YOLO11s-VV": "polisar_yolo_obb_larger_models/runs_obb_methods/YOLO11s-OBB_single_vv/results.csv",
    "YOLO11m-VV": "polisar_yolo_obb_larger_models/runs_obb_methods/YOLO11m-OBB_single_vv/results.csv",
}


def read_loss_csv(stream):
    rows = []
    reader = csv.DictReader(stream)
    for raw in reader:
        row = {key.strip(): value.strip() for key, value in raw.items() if key is not None}
        if "epoch" not in row:
            raise ValueError("Ultralytics results.csv has no epoch column")
        parsed = {"epoch": int(float(row["epoch"])) + 1}
        for key, value in row.items():
            if key.startswith(("train/", "val/")) and key.endswith("_loss") and value:
                parsed[key] = float(value)
        rows.append(parsed)
    return rows


def plot_model(name, rows, out):
    names = ("box_loss", "cls_loss", "dfl_loss", "angle_loss")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for axis, component in zip(axes.flat, names):
        for prefix, label, color in (("train", "Train", "#2b6cb0"),
                                     ("val", "Validation", "#c05621")):
            key = f"{prefix}/{component}"
            available = [row for row in rows if key in row]
            if available:
                axis.plot([row["epoch"] for row in available],
                          [row[key] for row in available], label=label, color=color)
        axis.set(title=component.replace("_", " ").title(), xlabel="Epoch", ylabel="Loss")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle(f"{name}: training and validation loss (results.csv)")
    fig.tight_layout()
    fig.savefig(out / f"{name.lower()}_exact_train_valid_loss.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/drdo"))
    parser.add_argument("--out", type=Path, default=Path("/content/drive/MyDrive/drdo/ppt_loss_curves"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    found = {}
    for name, relative in RUNS.items():
        path = args.drive_root / relative
        if not path.is_file():
            print("Missing:", path)
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            rows = read_loss_csv(stream)
        if not rows:
            print("Empty:", path)
            continue
        plot_model(name, rows, args.out)
        found[name] = rows
        print(f"Plotted {name}: {len(rows)} epochs")
    for names in (("YOLO11s-VV", "YOLO11m-VV"),
                  ("YOLO11n-VV", "YOLO11n-Pauli", "YOLO11n-LearnedFusion")):
        available = [name for name in names if name in found]
        if len(available) < 2:
            continue
        plt.figure(figsize=(8, 4.4))
        for name in available:
            rows = [row for row in found[name] if "val/box_loss" in row]
            plt.plot([r["epoch"] for r in rows], [r["val/box_loss"] for r in rows], label=name)
        plt.title("Validation box loss: " + " vs ".join(available))
        plt.xlabel("Epoch")
        plt.ylabel("Validation box loss")
        plt.grid(alpha=0.2)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(args.out / ("_vs_".join(name.lower() for name in available) + "_valid_box_loss.png"), dpi=220)
        plt.close()


if __name__ == "__main__":
    main()
