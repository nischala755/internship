"""Recover presentation plots from executed notebook text, without invented epochs."""

import argparse
import ast
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP = re.compile(
    r"(?P<name>[\w.]+) epoch (?P<epoch>\d+) (?P<phase>train|valid) "
    r"\[(?P<step>\d+)/(?P<total>\d+)\] loss=(?P<loss>[\d.]+)"
)
YOLO_TRAIN = re.compile(
    r"^\s*(?P<epoch>\d+)/(?P<total>\d+)\s+\S+\s+"
    r"(?P<box>[\d.]+)\s+(?P<cls>[\d.]+)\s+"
    r"(?P<dfl>[\d.]+)\s+(?P<angle>[\d.]+)\s+\d+\s+\d+: 100%"
)
YOLO_VALID = re.compile(
    r"^\s*all\s+\d+\s+\d+\s+(?P<p>[\d.]+)\s+"
    r"(?P<r>[\d.]+)\s+(?P<map50>[\d.]+)\s+(?P<map95>[\d.]+)\s*$"
)
FUSION = re.compile(
    r"fusion epoch (?P<epoch>\d+)/\d+ train_loss=(?P<train>[\d.]+) "
    r"valid_loss=(?P<valid>[\d.]+)"
)


def f1_from_pr(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def notebook_text(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(output.get("text", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    )


def parse_epoch_loss(log, name):
    by_epoch = {}
    for match in STEP.finditer(ANSI.sub("", log)):
        if match["name"] != name or match["step"] != match["total"]:
            continue
        epoch = int(match["epoch"])
        by_epoch.setdefault(epoch, {"epoch": epoch})[f"{match['phase']}_loss"] = float(match["loss"])
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def parse_yolo_epochs(log):
    rows = []
    current = None
    for line in ANSI.sub("", log).splitlines():
        train = YOLO_TRAIN.match(line)
        if train:
            current = {
                "epoch": int(train["epoch"]),
                "train_box_loss": float(train["box"]),
                "train_cls_loss": float(train["cls"]),
                "train_dfl_loss": float(train["dfl"]),
                "train_angle_loss": float(train["angle"]),
            }
            rows.append(current)
            continue
        valid = YOLO_VALID.match(line)
        if valid and current is not None and "valid_precision" not in current:
            p, r = float(valid["p"]), float(valid["r"])
            current.update({
                "valid_precision": p,
                "valid_recall": r,
                "valid_f1_harmonic": f1_from_pr(p, r),
                "valid_map50": float(valid["map50"]),
                "valid_map50_95": float(valid["map95"]),
            })
    return rows


def parse_fusion(log):
    return [
        {"epoch": int(m["epoch"]), "train_loss": float(m["train"]),
         "valid_loss": float(m["valid"])}
        for m in FUSION.finditer(ANSI.sub("", log))
    ]


def parse_resnet_history(log):
    """Extract complete ResNet epoch dictionaries printed by the notebook."""
    required = {
        "epoch", "train_loss", "train_accuracy", "train_f1_macro",
        "valid_loss", "valid_accuracy", "valid_f1_macro",
    }
    by_epoch = {}
    for match in re.finditer(r"\{[^{}]+\}", ANSI.sub("", log), flags=re.DOTALL):
        try:
            row = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if required <= row.keys():
            by_epoch[int(row["epoch"])] = {
                key: float(row[key]) if key != "epoch" else int(row[key])
                for key in required
            }
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def parse_nonyolo_epochs(log):
    methods = {}
    for line in ANSI.sub("", log).splitlines():
        if not line.startswith("{'method':"):
            continue
        try:
            raw = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if not {"method", "epoch", "loss", "precision", "recall", "f1"} <= raw.keys():
            continue
        row = {"epoch": int(raw["epoch"]), "train_loss": float(raw["loss"]),
               "valid_precision": float(raw["precision"]),
               "valid_recall": float(raw["recall"]), "valid_f1": float(raw["f1"])}
        methods.setdefault(raw["method"], {})[row["epoch"]] = row
    return {name: [epochs[key] for key in sorted(epochs)] for name, epochs in methods.items()}


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, title, ylabel):
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def draw_loss(path, rows, title):
    plt.figure(figsize=(8, 4.4))
    for key, label in (("train_loss", "Train"), ("valid_loss", "Validation")):
        valid = [r for r in rows if key in r]
        if valid:
            plt.plot([r["epoch"] for r in valid], [r[key] for r in valid],
                     label=label, linewidth=2)
    save_plot(path, title, "Loss")


def draw_yolo(path, rows, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r["epoch"] for r in rows]
    for key, label in (("train_box_loss", "Box"), ("train_cls_loss", "Class"),
                       ("train_dfl_loss", "DFL"), ("train_angle_loss", "Angle")):
        axes[0].plot(epochs, [r[key] for r in rows], label=label)
    axes[0].set(title="Training loss components", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(frameon=False)
    for key, label in (("valid_precision", "Precision"), ("valid_recall", "Recall"),
                       ("valid_f1_harmonic", "Harmonic P/R F1"),
                       ("valid_map50_95", "mAP50-95")):
        valid = [r for r in rows if key in r]
        axes[1].plot([r["epoch"] for r in valid], [r[key] for r in valid], label=label)
    axes[1].set(title="Validation detection metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(name + " | console log; validation loss not printed")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def draw_resnet_diagnostics(path, rows):
    epochs = [row["epoch"] for row in rows]
    best_valid = max(rows, key=lambda row: row["valid_accuracy"])
    final = rows[-1]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    axes[0].semilogy(epochs, [row["train_loss"] for row in rows], "o-", label="Train")
    axes[0].semilogy(epochs, [row["valid_loss"] for row in rows], "o-", label="Validation")
    axes[0].set(title="Cross-entropy loss (log scale)", ylabel="Loss")
    for axis, train_key, valid_key, title in (
        (axes[1], "train_accuracy", "valid_accuracy", "Classification accuracy"),
        (axes[2], "train_f1_macro", "valid_f1_macro", "Macro-F1"),
    ):
        axis.plot(epochs, [100 * row[train_key] for row in rows], "o-", label="Train")
        axis.plot(epochs, [100 * row[valid_key] for row in rows], "o-", label="Validation")
        axis.set(title=title, ylabel="Score (%)", ylim=(0, 105))
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.22)
        axis.legend(frameon=False)
    axes[1].scatter([best_valid["epoch"]], [100 * best_valid["valid_accuracy"]],
                    marker="*", s=150, color="#7a3e00", zorder=5,
                    label="Best validation")
    axes[1].annotate(
        f"Best validation: {100 * best_valid['valid_accuracy']:.2f}%\n(epoch {best_valid['epoch']})",
        (best_valid["epoch"], 100 * best_valid["valid_accuracy"]),
        xytext=(8, 10), textcoords="offset points", fontsize=8, color="#7a3e00")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Magnitude-only ResNet18 diagnostic: large train-validation gap",
                 fontsize=15, weight="bold")
    fig.text(.5, .01,
             f"Final epoch {int(final['epoch'])}: train accuracy {100 * final['train_accuracy']:.2f}%, "
             f"validation accuracy {100 * final['valid_accuracy']:.2f}%. "
             "Final-model held-out test: accuracy 41.36%, macro-F1 38.05%.\n"
             "Diagnostic run; training-only clipping makes it unsuitable as a controlled magnitude-vs-complex conclusion.",
             ha="center", fontsize=10, color="#874437")
    fig.tight_layout(rect=(0, .10, 1, .92))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "ppt_assets_verified" / "recovered_curves")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {
        "complex": "POLISAR_ComplexCNN_Classification_CLEAN (2).ipynb",
        "resnet": "POLISAR_Feature_Normalization_ResNet18_Study (1) (1).ipynb",
        "large": "POLISAR_YOLO_OBB_LargerModels_T4 (1).ipynb",
        "pauli": "POLISAR_OBB_Detection_PauliFusion_YOLO (1) (1).ipynb",
        "learned": "POLISAR_OBB_LearnableFusion_YOLO_Comparison (1) (1).ipynb",
        "nonyolo": "POLISAR_NonYOLO_OBB_PureTorch_Comparison_REVIEWED (1) (1).ipynb",
    }
    logs = {key: notebook_text(args.downloads / name) for key, name in sources.items()}
    manifest = {"sources": sources, "datasets": {}, "caveats": [
        "Console losses are rounded to four significant displayed decimals.",
        "YOLO validation loss is not in the exported console logs. Never infer it from mAP.",
        "F1 from logged aggregate precision/recall is a harmonic summary, not per-class macro F1.",
        "YOLO held-out test summaries used elsewhere are provisional due to 174 vs 162 duplicate-suffixed test labels.",
    ]}

    folds = {}
    for index in range(1, 6):
        name = f"fold_{index}"
        rows = parse_epoch_loss(logs["complex"], name)
        if rows:
            folds[name] = rows
            save_csv(args.out / f"complexcnn_{name}.csv", rows)
    if folds:
        plt.figure(figsize=(8, 4.4))
        for name, rows in folds.items():
            for key, color in (("train_loss", "#2b6cb0"), ("valid_loss", "#c05621")):
                present = [r for r in rows if key in r]
                plt.plot([r["epoch"] for r in present], [r[key] for r in present],
                         color=color, alpha=0.3, linewidth=1)
        for key, color, label in (("train_loss", "#2b6cb0", "Train mean"),
                                  ("valid_loss", "#c05621", "Validation mean")):
            epochs = sorted({r["epoch"] for rows in folds.values() for r in rows if key in r})
            means = [np.mean([r[key] for rows in folds.values() for r in rows
                              if r["epoch"] == epoch and key in r]) for epoch in epochs]
            plt.plot(epochs, means, color=color, linewidth=2.5, label=label)
        save_plot(args.out / "complexcnn_5fold_train_valid_loss.png",
                  "ComplexCNN: 5-fold loss (thin lines = individual folds)", "Classification loss")
        manifest["datasets"]["complexcnn_5fold"] = {"folds": {k: len(v) for k, v in folds.items()},
            "source": sources["complex"], "note": "Cross-validation models; not final refit."}

    resnet = parse_resnet_history(logs["resnet"])
    save_csv(args.out / "resnet18_train_valid_loss.csv", resnet)
    if resnet:
        draw_loss(args.out / "resnet18_train_valid_loss.png", resnet,
                  "ResNet18 magnitude: training vs validation loss")
        draw_resnet_diagnostics(args.out.parent / "guide_slide03_resnet18_training_diagnostics.png", resnet)
    manifest["datasets"]["resnet18"] = {"epochs": len(resnet), "source": sources["resnet"]}

    yolo_specs = (("large", "YOLO11s-VV", "YOLO11m-VV"),
                  ("pauli", "YOLO11n-VV", "YOLO11n-Pauli"),
                  ("learned", "YOLO11n-LearnedFusion",))
    extracted = {}
    for source, *names in yolo_specs:
        sections = logs[source].split("Starting training for ")[1:]
        for name, section in zip(names, sections):
            rows = parse_yolo_epochs(section)
            extracted[name] = rows
            save_csv(args.out / f"{name.lower()}_epochs.csv", rows)
            if rows:
                draw_yolo(args.out / f"{name.lower()}_train_loss_valid_metrics.png", rows, name)
            manifest["datasets"][name] = {"epochs": len(rows),
                "validation_metric_epochs": sum("valid_precision" in r for r in rows),
                "source": sources[source], "validation_loss": "not in console output"}

    for name, rows in parse_nonyolo_epochs(logs["nonyolo"]).items():
        save_csv(args.out / f"{name.lower()}_epochs.csv", rows)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        axes[0].plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows],
                     color="#2b6cb0", label="Train loss")
        axes[1].plot([r["epoch"] for r in rows], [r["valid_precision"] for r in rows], label="P")
        axes[1].plot([r["epoch"] for r in rows], [r["valid_recall"] for r in rows], label="R")
        axes[1].plot([r["epoch"] for r in rows], [r["valid_f1"] for r in rows], label="F1")
        for axis in axes:
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
        axes[0].set(title="Training loss", ylabel="Loss")
        metric_max = max(max(r["valid_precision"], r["valid_recall"], r["valid_f1"]) for r in rows)
        axes[1].set(title="Validation detection metrics", ylabel="Score",
                    ylim=(0, min(1, metric_max * 1.15)))
        fig.suptitle(name + " | validation loss not logged")
        fig.tight_layout()
        fig.savefig(args.out / f"{name.lower()}_train_loss_valid_metrics.png", dpi=220)
        plt.close(fig)
        manifest["datasets"][name] = {"epochs": len(rows), "source": sources["nonyolo"],
            "validation_loss": "not recorded in notebook output",
            "note": "Research prototypes, not reference implementations of full CenterNet or DETR."}

    first, second = extracted.get("YOLO11s-VV", []), extracted.get("YOLO11m-VV", [])
    if first and second:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
        for rows, label, color in ((first, "YOLO11s", "#2b6cb0"),
                                   (second, "YOLO11m", "#c05621")):
            valid = [r for r in rows if "valid_precision" in r]
            for axis, key in zip(axes, ("valid_precision", "valid_recall", "valid_f1_harmonic")):
                axis.plot([r["epoch"] for r in valid], [r[key] for r in valid],
                          label=label, color=color, linewidth=1.7)
        for axis, title in zip(axes, ("Precision", "Recall", "Harmonic P/R F1")):
            axis.set(title=title, xlabel="Epoch", ylim=(0, 1))
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Validation score")
        axes[0].legend(frameon=False)
        fig.suptitle("YOLO11s vs YOLO11m: per-epoch validation metrics")
        fig.tight_layout()
        fig.savefig(args.out / "yolo11s_vs_yolo11m_validation_prf1.png", dpi=220)
        plt.close(fig)

    fusion = parse_fusion(logs["learned"])
    save_csv(args.out / "fusion_stem_train_valid_loss.csv", fusion)
    if fusion:
        draw_loss(args.out / "fusion_stem_train_valid_loss.png", fusion,
                  "Learnable fusion stem: supervised BCE loss")
    manifest["datasets"]["fusion_stem"] = {"epochs": len(fusion), "source": sources["learned"],
        "note": "BCE against OBB-derived foreground/boundary masks and scattering energy; not detector loss."}
    (args.out / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["datasets"], indent=2))


if __name__ == "__main__":
    main()
