# %% [markdown]
# # Pol-ISAR ResNet18: imgaug, Shuffling, and Leakage-Aware 5-Fold Comparison
#
# This notebook compares the same four-channel magnitude ResNet18 with and
# without `imgaug` augmentation. It uses shuffled, azimuth-grouped five-fold
# cross-validation on the official train+validation development pool and reserves
# the official 162-image test set for final five-model ensemble evaluation.
#
# Every fold fits normalization using only that fold's training indices. Fold
# assignments, configurations, histories, checkpoints, predictions, and plots are
# saved to Drive so the full ten-run study can resume after a disconnection.

# %% [markdown]
# ## 1. Install imgaug without changing Colab's core packages
#
# `--no-deps` is intentional: Colab already supplies NumPy, SciPy, scikit-image,
# OpenCV, Pillow, and six. This prevents pip from downgrading unrelated packages.

# %%
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("imgaug") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "imgaug==0.4.0"])
print("imgaug package is available.")


# %% [markdown]
# ## 2. Runtime imports and NumPy compatibility
#
# imgaug 0.4.0 predates NumPy 2.x. These aliases restore names removed by NumPy
# without downgrading NumPy or altering PyTorch, Pillow, Requests, or Colab.

# %%
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

if "sctypes" not in np.__dict__:
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.object_, np.bytes_, np.str_],
    }
for alias, target in {"bool": np.bool_, "int": int, "float": float, "complex": complex, "object": object, "str": str}.items():
    if alias not in np.__dict__:
        setattr(np, alias, target)

import imgaug as ia
from imgaug import augmenters as iaa

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("scikit-learn:", sklearn.__version__)
print("imgaug:", ia.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# %% [markdown]
# ## 3. Mount Drive and freeze the experiment configuration
#
# The architecture is deliberately the same conventional baseline family as the
# earlier experiment: ResNet18 trained from scratch, with its first convolution
# changed from 3 to 4 channels and its final layer changed to 6 classes.

# %%
try:
    from google.colab import drive
except ImportError:
    print("Google Drive mount skipped outside Colab.")
else:
    # MyDrive can exist as a stale empty directory after a runtime restart. The
    # project folder is a stronger mount check than the mountpoint alone.
    if not Path("/content/drive/MyDrive/drdo").is_dir():
        print("MyDrive/drdo is unavailable; forcing a clean Drive remount.")
        drive.mount("/content/drive", force_remount=True)
    else:
        print("Drive is mounted and MyDrive/drdo is visible.")


@dataclass(frozen=True)
class Config:
    data_root: str = "/content/drive/MyDrive/drdo/Pol_ISARSpaceTarget1.0"
    output_dir: str = "/content/drive/MyDrive/drdo/outputs/resnet18_imgaug_shuffled_5fold"
    persistent_cache_dir: str = "/content/drive/MyDrive/drdo/cache/resnet18_mag_log1p_224_v1"
    local_cache_dir: str = "/content/resnet18_mag_log1p_224_v1"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 100
    n_splits: int = 5
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 1.0
    augmentation_degrees: float = 5.0
    augmentation_translate: float = 0.03
    augmentation_scale_min: float = 0.95
    augmentation_scale_max: float = 1.05
    augmentation_noise_std_min: float = 0.0
    augmentation_noise_std_max: float = 0.02
    seed: int = 42
    expected_train: int = 1314
    expected_valid: int = 162
    expected_test: int = 162
    strict_counts: bool = True


cfg = Config()

# Resolve both the normal Drive layout and an accidental extra extraction level.
def has_official_split_layout(root: Path) -> bool:
    return all((root / "ImageSets" / "images" / split).is_dir() for split in ("train", "valid", "test"))


def resolve_dataset_root(preferred: Path) -> Path:
    candidates = [preferred, Path.cwd() / "Pol_ISARSpaceTarget1.0"]
    drive_project = Path("/content/drive/MyDrive/drdo")
    if drive_project.is_dir():
        # Covers MyDrive/drdo/<dataset>/ImageSets/images/train and one extra
        # nested extraction directory without recursively scanning every .dat.
        train_patterns = (
            "*/ImageSets/images/train",
            "*/*/ImageSets/images/train",
        )
        for pattern in train_patterns:
            for train_dir in drive_project.glob(pattern):
                candidates.append(train_dir.parents[2])

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if has_official_split_layout(candidate):
            return candidate

    my_drive = Path("/content/drive/MyDrive")
    available = sorted(path.name for path in my_drive.iterdir()) if my_drive.is_dir() else []
    raise FileNotFoundError(
        "Could not locate a complete Pol-ISAR dataset. Required relative path: "
        "ImageSets/images/train, valid, and test. "
        f"Checked: {[str(path) for path in checked]}. "
        f"Available entries under MyDrive: {available[:30]}. "
        "Confirm that Colab is mounted to the Google account containing MyDrive/drdo."
    )


resolved_data_root = resolve_dataset_root(Path(cfg.data_root))
object.__setattr__(cfg, "data_root", str(resolved_data_root))

# Permit local verification without changing Drive output behavior in Colab.
if not str(resolved_data_root).startswith("/content/drive/"):
    object.__setattr__(cfg, "output_dir", str((Path.cwd() / "outputs" / "resnet18_imgaug_shuffled_5fold").resolve()))
    object.__setattr__(cfg, "persistent_cache_dir", str((Path.cwd() / "cache" / "resnet18_mag_log1p_224_v1").resolve()))
    object.__setattr__(cfg, "local_cache_dir", str((Path.cwd() / "cache" / "resnet18_mag_log1p_224_v1_local").resolve()))

DATA_ROOT = Path(cfg.data_root)
OUT_DIR = Path(cfg.output_dir)
PERSISTENT_CACHE = Path(cfg.persistent_cache_dir)
LOCAL_CACHE = Path(cfg.local_cache_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
PERSISTENT_CACHE.mkdir(parents=True, exist_ok=True)
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

(OUT_DIR / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
print("Resolved DATA_ROOT:", DATA_ROOT)
print(json.dumps(asdict(cfg), indent=2))


# %% [markdown]
# ## 4. Reproducibility controls

# %%
def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(cfg.seed)
ia.seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_enabled = device.type == "cuda"
print("Device:", device, "| mixed precision:", amp_enabled)
if device.type != "cuda":
    print("WARNING: A GPU is strongly recommended for the full 100-epoch run.")


# %% [markdown]
# ## 5. Dataset discovery and leakage audit
#
# Every usable sample must have exactly four complete complex `.dat` channels.
# Google Drive duplicate copies such as `file (1).dat` are ignored. The notebook
# stops if counts differ from 1314/162/162, a class is absent, or an exact sample
# basename appears in more than one split.

# %%
POLS = ("HH", "HV", "VH", "VV")
CLASS_NAMES = ("Capella", "HISEA1", "QPSSAR", "Radarsat2", "StarLink", "Worldview4")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
DUPLICATE_RE = re.compile(r" \(\d+\)(?=\.[^.]+$)")
FRAME_RE = re.compile(r"^(?P<class_name>.+?)_el_(?P<elevation>-?\d+)_az_(?P<azimuth>-?\d+)$")


def infer_class_id(base: str) -> int:
    for name, class_id in CLASS_TO_ID.items():
        if base.startswith(name + "_"):
            return class_id
    raise ValueError(f"Cannot infer satellite class from {base!r}")


def parse_geometry(base: str) -> Tuple[int, int]:
    match = FRAME_RE.fullmatch(base)
    if match is None:
        raise ValueError(f"Cannot parse elevation/azimuth from {base!r}")
    return int(match.group("elevation")), int(match.group("azimuth"))


def discover_split(data_root: Path, split: str) -> pd.DataFrame:
    image_dir = data_root / "ImageSets" / "images" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {image_dir}")
    bases = sorted(
        {
            path.name.rsplit("_", 1)[0]
            for path in image_dir.glob("*_HH.dat")
            if not DUPLICATE_RE.search(path.name)
        }
    )
    rows = []
    incomplete = []
    for base in bases:
        paths = [image_dir / f"{base}_{pol}.dat" for pol in POLS]
        if not all(path.exists() for path in paths):
            incomplete.append(base)
            continue
        elevation, azimuth = parse_geometry(base)
        rows.append(
            {
                "split": split,
                "base": base,
                "class_id": infer_class_id(base),
                "class_name": CLASS_NAMES[infer_class_id(base)],
                "elevation": elevation,
                "azimuth": azimuth,
            }
        )
    if incomplete:
        raise RuntimeError(f"{split}: {len(incomplete)} incomplete full-pol samples; first five: {incomplete[:5]}")
    return pd.DataFrame(rows)


def assert_no_split_overlap(manifests: Dict[str, pd.DataFrame]) -> None:
    split_names = list(manifests)
    for left_idx, left in enumerate(split_names):
        for right in split_names[left_idx + 1 :]:
            overlap = set(manifests[left]["base"]) & set(manifests[right]["base"])
            if overlap:
                raise AssertionError(f"Data leakage: {len(overlap)} basenames overlap between {left} and {right}: {sorted(overlap)[:5]}")


manifests = {split: discover_split(DATA_ROOT, split) for split in ("train", "valid", "test")}
expected_counts = {"train": cfg.expected_train, "valid": cfg.expected_valid, "test": cfg.expected_test}
actual_counts = {split: len(frame) for split, frame in manifests.items()}
print("Split counts:", actual_counts)
if cfg.strict_counts and actual_counts != expected_counts:
    raise AssertionError(f"Expected {expected_counts}, found {actual_counts}. Stop and repair the dataset before training.")

assert_no_split_overlap(manifests)
for split, frame in manifests.items():
    missing_classes = sorted(set(CLASS_NAMES) - set(frame["class_name"]))
    if missing_classes:
        raise AssertionError(f"{split} is missing classes: {missing_classes}")

manifest = pd.concat(manifests.values(), ignore_index=True)
manifest.to_csv(OUT_DIR / "split_manifest.csv", index=False)
class_balance = pd.crosstab(manifest["class_name"], manifest["split"]).reindex(index=CLASS_NAMES, columns=["train", "valid", "test"])
class_balance.to_csv(OUT_DIR / "class_balance.csv")
display(class_balance)
print("Leakage audit passed: no exact basename overlap across official splits.")


# %% [markdown]
# ## 6. Build a persistent, RAM-safe preprocessing cache
#
# For each channel, the notebook reads the complex data, computes magnitude,
# applies `log(1 + magnitude)` to compress the SAR dynamic range, and resizes to
# 224 x 224. Arrays are written as float16 memory-mapped files, so the entire
# dataset is never held in RAM. A persistent Drive copy is reused after a runtime
# restart; training reads a local `/content` copy for speed.

# %%
def read_complex_channel(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    expected = 256 * 256 * 2
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} float32 values; expected {expected}")
    return values[0::2].reshape(256, 256) + 1j * values[1::2].reshape(256, 256)


def preprocess_sample(image_dir: Path, base: str) -> np.ndarray:
    channels = []
    for pol in POLS:
        complex_channel = read_complex_channel(image_dir / f"{base}_{pol}.dat")
        channels.append(np.log1p(np.abs(complex_channel)).astype(np.float32))
    tensor = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).numpy().astype(np.float16)


def cache_metadata(split: str, frame: pd.DataFrame) -> Dict[str, object]:
    return {
        "split": split,
        "count": len(frame),
        "shape": [len(frame), 4, cfg.image_size, cfg.image_size],
        "dtype": "float16",
        "preprocessing": "log1p(abs(complex)), bilinear resize",
        "image_size": cfg.image_size,
        "bases_sha256": hashlib.sha256("\n".join(frame["base"]).encode("utf-8")).hexdigest(),
    }


def validate_npy_array(path: Path, expected_shape: Tuple[int, ...], expected_dtype: np.dtype) -> bool:
    """Reject missing, empty, truncated, malformed, or incompatible NPY files."""
    if not path.is_file() or path.stat().st_size < 128:
        return False
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        valid = tuple(array.shape) == tuple(expected_shape) and array.dtype == np.dtype(expected_dtype)
        del array
        return bool(valid)
    except (EOFError, OSError, ValueError):
        return False


def persistent_cache_is_valid(split: str, frame: pd.DataFrame) -> bool:
    array_path = PERSISTENT_CACHE / f"{split}_x.npy"
    labels_path = PERSISTENT_CACHE / f"{split}_y.npy"
    bases_path = PERSISTENT_CACHE / f"{split}_bases.json"
    metadata_path = PERSISTENT_CACHE / f"{split}_metadata.json"
    expected_x_shape = (len(frame), 4, cfg.image_size, cfg.image_size)
    if not validate_npy_array(array_path, expected_x_shape, np.float16):
        return False
    if not validate_npy_array(labels_path, (len(frame),), np.int64):
        return False
    if not (bases_path.is_file() and metadata_path.is_file()):
        return False
    try:
        metadata_matches = json.loads(metadata_path.read_text(encoding="utf-8")) == cache_metadata(split, frame)
        bases_match = json.loads(bases_path.read_text(encoding="utf-8")) == frame["base"].tolist()
        return metadata_matches and bases_match
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def build_persistent_cache(split: str, frame: pd.DataFrame) -> None:
    if persistent_cache_is_valid(split, frame):
        print(f"{split}: valid persistent cache already exists")
        return
    stale_paths = [
        PERSISTENT_CACHE / f"{split}_x.npy",
        PERSISTENT_CACHE / f"{split}_y.npy",
        PERSISTENT_CACHE / f"{split}_bases.json",
        PERSISTENT_CACHE / f"{split}_metadata.json",
        PERSISTENT_CACHE / f"{split}_x.building.npy",
        PERSISTENT_CACHE / f"{split}_y.building.npy",
    ]
    if any(path.exists() for path in stale_paths):
        print(f"{split}: Corrupt persistent cache detected; rebuilding this split.")
    for path in stale_paths:
        path.unlink(missing_ok=True)

    array_path = PERSISTENT_CACHE / f"{split}_x.npy"
    tmp_path = PERSISTENT_CACHE / f"{split}_x.building.npy"
    mmap = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(frame), 4, cfg.image_size, cfg.image_size),
    )
    image_dir = DATA_ROOT / "ImageSets" / "images" / split
    for idx, base in enumerate(frame["base"]):
        mmap[idx] = preprocess_sample(image_dir, base)
        if idx == 0 or (idx + 1) % 100 == 0 or idx + 1 == len(frame):
            print(f"{split} cache: {idx + 1}/{len(frame)}")
    mmap.flush()
    del mmap
    tmp_path.replace(array_path)
    labels_tmp = PERSISTENT_CACHE / f"{split}_y.building.npy"
    np.save(labels_tmp, frame["class_id"].to_numpy(dtype=np.int64))
    labels_tmp.replace(PERSISTENT_CACHE / f"{split}_y.npy")
    (PERSISTENT_CACHE / f"{split}_bases.json").write_text(json.dumps(frame["base"].tolist(), indent=2), encoding="utf-8")
    # Metadata is intentionally written last: its presence means the cache build
    # reached completion, but structural NPY validation is still required.
    (PERSISTENT_CACHE / f"{split}_metadata.json").write_text(json.dumps(cache_metadata(split, frame), indent=2), encoding="utf-8")
    if not persistent_cache_is_valid(split, frame):
        raise RuntimeError(f"{split}: persistent cache failed post-build integrity validation")


for split, frame in manifests.items():
    build_persistent_cache(split, frame)


def local_cache_is_valid(split: str, frame: pd.DataFrame) -> bool:
    if not validate_npy_array(LOCAL_CACHE / f"{split}_x.npy", (len(frame), 4, cfg.image_size, cfg.image_size), np.float16):
        return False
    if not validate_npy_array(LOCAL_CACHE / f"{split}_y.npy", (len(frame),), np.int64):
        return False
    try:
        metadata_matches = json.loads((LOCAL_CACHE / f"{split}_metadata.json").read_text(encoding="utf-8")) == cache_metadata(split, frame)
        bases_match = json.loads((LOCAL_CACHE / f"{split}_bases.json").read_text(encoding="utf-8")) == frame["base"].tolist()
        return metadata_matches and bases_match
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def copy_cache_to_local(split: str, frame: pd.DataFrame) -> None:
    if local_cache_is_valid(split, frame):
        print(f"{split}: valid local runtime cache already exists")
        return
    if any((LOCAL_CACHE / f"{split}_{suffix}").exists() for suffix in ("x.npy", "y.npy", "bases.json", "metadata.json")):
        print(f"{split}: corrupt or incomplete local cache detected; recopying atomically.")
    for suffix in ("x.npy", "y.npy", "bases.json", "metadata.json"):
        source = PERSISTENT_CACHE / f"{split}_{suffix}"
        destination = LOCAL_CACHE / source.name
        temporary = destination.with_suffix(destination.suffix + ".copying")
        temporary.unlink(missing_ok=True)
        print("Copying to local runtime:", source.name)
        shutil.copy2(source, temporary)
        if suffix == "x.npy" and not validate_npy_array(temporary, (len(frame), 4, cfg.image_size, cfg.image_size), np.float16):
            raise RuntimeError(f"Local copy validation failed for {source.name}")
        if suffix == "y.npy" and not validate_npy_array(temporary, (len(frame),), np.int64):
            raise RuntimeError(f"Local copy validation failed for {source.name}")
        temporary.replace(destination)
    if not local_cache_is_valid(split, frame):
        raise RuntimeError(f"{split}: local cache failed post-copy integrity validation")


for split, frame in manifests.items():
    copy_cache_to_local(split, frame)


# %% [markdown]
# ## 7. Development pool and shuffled, azimuth-grouped five folds
#
# The official train and validation splits form the development pool. The
# official test split remains untouched until the final ensemble section.
# `StratifiedGroupKFold` is shuffled with seed 42 while grouping by azimuth;
# therefore, all classes/elevations at one azimuth stay in one fold.

# %%
development_parts = []
for source_split in ("train", "valid"):
    part = manifests[source_split].copy().reset_index(drop=True)
    part["source_split"] = source_split
    part["source_index"] = np.arange(len(part), dtype=np.int64)
    development_parts.append(part)
development_manifest = pd.concat(development_parts, ignore_index=True)
test_manifest = manifests["test"].copy().reset_index(drop=True)
test_manifest["source_split"] = "test"
test_manifest["source_index"] = np.arange(len(test_manifest), dtype=np.int64)

splitter = StratifiedGroupKFold(
    n_splits=cfg.n_splits,
    shuffle=True,
    random_state=cfg.seed,
)
fold_indices = []
development_manifest["validation_fold"] = -1
for fold, (train_indices, validation_indices) in enumerate(
    splitter.split(
        X=np.zeros(len(development_manifest)),
        y=development_manifest["class_id"],
        groups=development_manifest["azimuth"],
    ),
    start=1,
):
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    train_groups = set(development_manifest.iloc[train_indices]["azimuth"])
    validation_groups = set(development_manifest.iloc[validation_indices]["azimuth"])
    assert train_groups.isdisjoint(validation_groups)
    assert set(train_indices).isdisjoint(set(validation_indices))
    fold_indices.append((train_indices, validation_indices))
    development_manifest.loc[validation_indices, "validation_fold"] = fold
    train_counts = development_manifest.iloc[train_indices]["class_name"].value_counts().reindex(CLASS_NAMES)
    validation_counts = development_manifest.iloc[validation_indices]["class_name"].value_counts().reindex(CLASS_NAMES)
    print(
        f"Fold {fold}: train={len(train_indices)}, validation={len(validation_indices)}, "
        f"train azimuths={len(train_groups)}, validation azimuths={len(validation_groups)}"
    )
    display(pd.DataFrame({"train": train_counts, "validation": validation_counts}))

assert (development_manifest["validation_fold"] > 0).all()
development_manifest.to_csv(OUT_DIR / "development_fold_assignments.csv", index=False)
test_manifest.to_csv(OUT_DIR / "untouched_official_test_manifest.csv", index=False)
print("Development samples:", len(development_manifest), "| untouched test samples:", len(test_manifest))


# %% [markdown]
# ## 8. Cached sample store and fold-training-only normalization

# %%
class CachedSampleStore:
    def __init__(self) -> None:
        self.arrays = {split: np.load(LOCAL_CACHE / f"{split}_x.npy", mmap_mode="r") for split in manifests}
        self.labels = {split: np.load(LOCAL_CACHE / f"{split}_y.npy", mmap_mode="r") for split in manifests}

    def sample(self, row: pd.Series) -> np.ndarray:
        return np.array(self.arrays[row["source_split"]][int(row["source_index"])], dtype=np.float32, copy=True)


store = CachedSampleStore()


def compute_fold_train_stats(indices: Sequence[int], chunk_size: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    channel_sum = np.zeros(4, dtype=np.float64)
    channel_sumsq = np.zeros(4, dtype=np.float64)
    pixel_count = 0
    indices = list(map(int, indices))
    for start in range(0, len(indices), chunk_size):
        batch_indices = indices[start : start + chunk_size]
        chunk = np.stack([store.sample(development_manifest.iloc[index]) for index in batch_indices])
        channel_sum += chunk.sum(axis=(0, 2, 3), dtype=np.float64)
        channel_sumsq += np.square(chunk, dtype=np.float32).sum(axis=(0, 2, 3), dtype=np.float64)
        pixel_count += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_sumsq / pixel_count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


fold_stats = {}
for fold, (train_indices, _) in enumerate(fold_indices, start=1):
    stats_path = OUT_DIR / f"fold_{fold}" / "fold_train_stats.npz"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if stats_path.exists() and np.array_equal(np.load(stats_path)["train_indices"], train_indices):
        saved = np.load(stats_path)
        mean, std = saved["mean"], saved["std"]
    else:
        mean, std = compute_fold_train_stats(train_indices)
        np.savez(stats_path, mean=mean, std=std, train_indices=train_indices)
    fold_stats[fold] = (mean, std)
    print(f"Fold {fold} training-only mean={mean.tolist()} std={std.tolist()}")


# %% [markdown]
# ## 9. imgaug pipeline and fold datasets
#
# The four channels are converted from CHW to HWC and augmented jointly, so one
# geometric transform is applied consistently to HH, HV, VH, and VV. RGB-only
# color augmenters, arbitrary flips, and 90-degree rotations are excluded.

# %%
REGIMES = ("no_augmentation", "imgaug")


def build_imgaug_pipeline() -> iaa.Sequential:
    return iaa.Sequential(
        [
            iaa.Sometimes(
                0.85,
                iaa.Affine(
                    scale=(cfg.augmentation_scale_min, cfg.augmentation_scale_max),
                    translate_percent={"x": (-cfg.augmentation_translate, cfg.augmentation_translate), "y": (-cfg.augmentation_translate, cfg.augmentation_translate)},
                    rotate=(-cfg.augmentation_degrees, cfg.augmentation_degrees),
                    order=1,
                    mode="constant",
                    cval=0.0,
                ),
            ),
            iaa.Sometimes(0.30, iaa.GaussianBlur(sigma=(0.0, 0.4))),
            iaa.Sometimes(0.70, iaa.Multiply((0.98, 1.02), per_channel=False)),
            iaa.AdditiveGaussianNoise(
                loc=0.0,
                scale=(cfg.augmentation_noise_std_min, cfg.augmentation_noise_std_max),
                per_channel=False,
            ),
        ],
        random_order=False,
    )


class FoldDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        indices: Sequence[int],
        mean: np.ndarray,
        std: np.ndarray,
        use_imgaug: bool,
    ) -> None:
        self.manifest = manifest
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(4, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(4, 1, 1)
        self.use_imgaug = use_imgaug
        self.augmenter = build_imgaug_pipeline() if use_imgaug else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Dict[str, object]:
        row = self.manifest.iloc[int(self.indices[position])]
        x = store.sample(row)
        if self.augmenter is not None:
            hwc = np.moveaxis(x, 0, -1)
            hwc = self.augmenter(image=hwc)
            x = np.moveaxis(np.asarray(hwc, dtype=np.float32), -1, 0)
            x = np.maximum(x, 0.0)
        tensor = torch.from_numpy(np.ascontiguousarray(x))
        tensor = (tensor - self.mean) / self.std
        return {"x": tensor, "y": torch.tensor(int(row["class_id"]), dtype=torch.long), "base": str(row["base"])}


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    ia.seed(worker_seed)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and getattr(worker_info.dataset, "augmenter", None) is not None:
        worker_info.dataset.augmenter.seed_(worker_seed)


def make_loader(dataset: Dataset, shuffle: bool, seed: int) -> Tuple[DataLoader, torch.Generator]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=amp_enabled,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, generator


# Visual audit of the exact joint transform applied to all polarizations.
preview_row = development_manifest.iloc[int(fold_indices[0][0][0])]
preview_original = store.sample(preview_row)
preview_hwc = np.moveaxis(preview_original, 0, -1)
preview_augmented = np.moveaxis(
    np.maximum(build_imgaug_pipeline()(image=preview_hwc), 0.0).astype(np.float32),
    -1,
    0,
)
figure, axes = plt.subplots(2, 4, figsize=(14, 7))
for channel, polarization in enumerate(POLS):
    vmax = float(np.percentile(preview_original[channel], 99.5))
    axes[0, channel].imshow(preview_original[channel], cmap="gray", vmin=0, vmax=vmax)
    axes[0, channel].set_title(f"Original {polarization}")
    axes[1, channel].imshow(preview_augmented[channel], cmap="gray", vmin=0, vmax=vmax)
    axes[1, channel].set_title(f"imgaug {polarization}")
    axes[0, channel].axis("off")
    axes[1, channel].axis("off")
figure.suptitle(f"Joint four-channel augmentation audit: {preview_row['base']}", fontweight="bold")
figure.tight_layout()
figure.savefig(OUT_DIR / "augmentation_preview.png", dpi=200, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 10. ResNet18, metrics, and training utilities

# %%
def make_resnet18() -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def metric_dict(targets: Sequence[int], predictions: Sequence[int], mean_loss: float) -> Dict[str, float]:
    return {
        "loss": float(mean_loss),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision_macro": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }


def run_epoch(model, loader, criterion, optimizer=None, scaler=None) -> Tuple[Dict[str, float], Dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
    targets, predictions, probabilities, bases = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
        batch_size = y.size(0)
        loss_sum += float(loss.detach()) * batch_size
        sample_count += batch_size
        probs = torch.softmax(logits.detach(), dim=1).cpu()
        targets.extend(y.cpu().tolist())
        predictions.extend(probs.argmax(dim=1).tolist())
        probabilities.extend(probs.tolist())
        bases.extend(list(batch["base"]))
    return metric_dict(targets, predictions, loss_sum / sample_count), {
        "targets": targets,
        "predictions": predictions,
        "probabilities": probabilities,
        "bases": bases,
    }


def atomic_torch_save(payload: Dict[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)


# %% [markdown]
# ## 11. Run two matched 5-fold, 100-epoch experiments
#
# Each `last_checkpoint.pt` is written after every epoch. Re-running this cell
# resumes incomplete folds and skips completed folds.

# %%
def train_one_fold(regime: str, fold: int, train_indices: np.ndarray, validation_indices: np.ndarray) -> Dict[str, object]:
    run_seed = cfg.seed + fold * 1000
    seed_everything(run_seed)
    ia.seed(run_seed)
    run_dir = OUT_DIR / regime / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    mean, std = fold_stats[fold]
    train_dataset = FoldDataset(development_manifest, train_indices, mean, std, use_imgaug=regime == "imgaug")
    validation_dataset = FoldDataset(development_manifest, validation_indices, mean, std, use_imgaug=False)
    train_loader, train_generator = make_loader(train_dataset, shuffle=True, seed=run_seed)
    validation_loader, _ = make_loader(validation_dataset, shuffle=False, seed=run_seed + 1)

    model = make_resnet18().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.minimum_learning_rate)
    scaler = make_scaler()
    last_checkpoint = run_dir / "last_checkpoint.pt"
    best_checkpoint = run_dir / "best_validation_macro_f1.pt"
    history_path = run_dir / "history.csv"
    history = []
    start_epoch = 0
    best_key = (-math.inf, math.inf)

    if last_checkpoint.exists():
        saved = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        # Loading the checkpoint onto CUDA also maps this RNG tensor to CUDA,
        # while torch.Generator.set_state requires a CPU ByteTensor.
        train_generator.set_state(saved["train_generator_state"].cpu())
        history = saved["history"]
        start_epoch = int(saved["epoch"])
        best_key = tuple(saved["best_key"])
        print(f"{regime} fold {fold}: resuming after epoch {start_epoch}")

    for epoch_index in range(start_epoch, cfg.epochs):
        epoch_number = epoch_index + 1
        started = time.time()
        train_metrics, _ = run_epoch(model, train_loader, criterion, optimizer=optimizer, scaler=scaler)
        validation_metrics, _ = run_epoch(model, validation_loader, criterion)
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        row = {
            "regime": regime,
            "fold": fold,
            "epoch": epoch_number,
            "learning_rate": current_lr,
            "seconds": time.time() - started,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"valid_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        selection_key = (validation_metrics["f1_macro"], -validation_metrics["loss"])
        payload = {
            "epoch": epoch_number,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "best_key": max(best_key, selection_key),
            "train_generator_state": train_generator.get_state(),
            "config": asdict(cfg),
            "regime": regime,
            "fold": fold,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
        }
        if selection_key > best_key:
            best_key = selection_key
            payload["best_key"] = best_key
            atomic_torch_save(payload, best_checkpoint)
        atomic_torch_save(payload, last_checkpoint)
        print(
            f"{regime} fold {fold} epoch {epoch_number:03d}/{cfg.epochs} | "
            f"train loss={train_metrics['loss']:.4f} F1={train_metrics['f1_macro']:.4f} | "
            f"valid loss={validation_metrics['loss']:.4f} F1={validation_metrics['f1_macro']:.4f} | "
            f"{row['seconds'] / 60:.1f} min"
        )

    assert len(history) == cfg.epochs
    best = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    validation_metrics, validation_outputs = run_epoch(model, validation_loader, criterion)
    prediction_frame = pd.DataFrame({
        "base": validation_outputs["bases"],
        "true_id": validation_outputs["targets"],
        "predicted_id": validation_outputs["predictions"],
    })
    prediction_frame.to_csv(run_dir / "validation_predictions.csv", index=False)
    summary = {"regime": regime, "fold": fold, "selected_epoch": int(best["epoch"]), **{f"valid_{k}": v for k, v in validation_metrics.items()}}
    (run_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del model, optimizer, scheduler, scaler, train_loader, validation_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


fold_summaries = []
for regime in REGIMES:
    for fold, (train_indices, validation_indices) in enumerate(fold_indices, start=1):
        fold_summaries.append(train_one_fold(regime, fold, train_indices, validation_indices))

fold_summary_df = pd.DataFrame(fold_summaries)
fold_summary_df.to_csv(OUT_DIR / "fold_validation_results.csv", index=False)
display(fold_summary_df)


# %% [markdown]
# ## 12. Cross-validation summaries and training curves

# %%
metric_columns = ["valid_accuracy", "valid_precision_macro", "valid_recall_macro", "valid_f1_macro", "valid_loss"]
summary_rows = []
for regime, group in fold_summary_df.groupby("regime"):
    row = {"regime": regime}
    for column in metric_columns:
        row[f"{column}_mean"] = float(group[column].mean())
        row[f"{column}_std"] = float(group[column].std(ddof=1))
    summary_rows.append(row)
cross_validation_summary = pd.DataFrame(summary_rows)
cross_validation_summary.to_csv(OUT_DIR / "cross_validation_summary.csv", index=False)
display(cross_validation_summary)

all_histories = []
for regime in REGIMES:
    for fold in range(1, cfg.n_splits + 1):
        all_histories.append(pd.read_csv(OUT_DIR / regime / f"fold_{fold}" / "history.csv"))
history_df = pd.concat(all_histories, ignore_index=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for regime in REGIMES:
    grouped = history_df[history_df["regime"] == regime].groupby("epoch")
    for axis, column, title in (
        (axes[0, 0], "train_loss", "Mean training loss"),
        (axes[0, 1], "valid_loss", "Mean validation loss"),
        (axes[1, 0], "valid_accuracy", "Mean validation accuracy"),
        (axes[1, 1], "valid_f1_macro", "Mean validation macro-F1"),
    ):
        mean = grouped[column].mean()
        std = grouped[column].std(ddof=1).fillna(0)
        axis.plot(mean.index, mean, label=regime)
        axis.fill_between(mean.index, mean - std, mean + std, alpha=0.15)
        axis.set(title=title, xlabel="Epoch", ylabel=column)
        axis.grid(alpha=0.25)
        axis.legend()
fig.suptitle("ResNet18 5-fold learning curves: no augmentation vs imgaug", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "five_fold_learning_curves.png", dpi=200, bbox_inches="tight")
plt.show()

plot_frame = fold_summary_df.pivot(index="fold", columns="regime", values="valid_f1_macro")
ax = plot_frame.plot(kind="bar", figsize=(10, 5), ylim=(0, 1.02))
ax.set(title="Best-checkpoint validation macro-F1 by fold", ylabel="Macro-F1", xlabel="Fold")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "augmentation_comparison.png", dpi=200, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Final ensemble evaluation on the untouched official test set
#
# Each fold model uses its own fold-training normalization statistics. The five
# probability matrices are averaged, then one ensemble prediction is produced
# for each official test image. The test set was not used for fold generation,
# augmentation selection, normalization fitting, or checkpoint selection.

# %%
def evaluate_ensemble_on_test(regime: str) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    ensemble_probabilities = []
    expected_bases = None
    targets = None
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    for fold in range(1, cfg.n_splits + 1):
        mean, std = fold_stats[fold]
        dataset = FoldDataset(test_manifest, np.arange(len(test_manifest)), mean, std, use_imgaug=False)
        loader, _ = make_loader(dataset, shuffle=False, seed=cfg.seed + fold)
        model = make_resnet18().to(device)
        checkpoint = torch.load(OUT_DIR / regime / f"fold_{fold}" / "best_validation_macro_f1.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        _, outputs = run_epoch(model, loader, criterion)
        if expected_bases is None:
            expected_bases = outputs["bases"]
            targets = np.asarray(outputs["targets"], dtype=np.int64)
        else:
            assert outputs["bases"] == expected_bases
            assert np.array_equal(np.asarray(outputs["targets"]), targets)
        ensemble_probabilities.append(np.asarray(outputs["probabilities"], dtype=np.float64))
        del model, loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_probabilities = np.mean(np.stack(ensemble_probabilities, axis=0), axis=0)
    predictions = mean_probabilities.argmax(axis=1)
    log_probabilities = np.log(np.clip(mean_probabilities, 1e-12, 1.0))
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[targets]
    smoothed_targets = (1.0 - cfg.label_smoothing) * one_hot + cfg.label_smoothing / len(CLASS_NAMES)
    ensemble_loss = float(-(smoothed_targets * log_probabilities).sum(axis=1).mean())
    metrics = metric_dict(targets, predictions, ensemble_loss)
    predictions_df = pd.DataFrame({
        "base": expected_bases,
        "true_id": targets,
        "true_class": [CLASS_NAMES[index] for index in targets],
        "predicted_id": predictions,
        "predicted_class": [CLASS_NAMES[index] for index in predictions],
    })
    for class_id, class_name in enumerate(CLASS_NAMES):
        predictions_df[f"prob_{class_name}"] = mean_probabilities[:, class_id]
    return metrics, predictions_df, mean_probabilities


def save_confusion(matrix: np.ndarray, title: str, destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(xticks=range(6), yticks=range(6), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted class", ylabel="True class", title=title)
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2
    for row in range(6):
        for column in range(6):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > threshold else "black")
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.show()


test_rows = []
for regime in REGIMES:
    metrics, predictions_df, _ = evaluate_ensemble_on_test(regime)
    predictions_df.to_csv(OUT_DIR / f"{regime}_test_ensemble_predictions.csv", index=False)
    report = classification_report(
        predictions_df["true_id"], predictions_df["predicted_id"],
        labels=range(6), target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(OUT_DIR / f"{regime}_test_classification_report.csv")
    matrix = confusion_matrix(predictions_df["true_id"], predictions_df["predicted_id"], labels=range(6))
    np.save(OUT_DIR / f"{regime}_test_confusion_matrix.npy", matrix)
    save_confusion(matrix, f"{regime}: five-fold ensemble on official test", OUT_DIR / f"{regime}_test_confusion_matrix.png")
    test_rows.append({"regime": regime, **{f"test_{key}": value for key, value in metrics.items()}})

test_comparison = pd.DataFrame(test_rows)
test_comparison.to_csv(OUT_DIR / "official_test_ensemble_comparison.csv", index=False)
display(test_comparison)

ax = test_comparison.set_index("regime")[["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]].plot(kind="bar", figsize=(10, 5), ylim=(0, 1.02))
ax.set(title="Official-test five-fold ensemble comparison", ylabel="Score", xlabel="Regime")
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "official_test_ensemble_comparison.png", dpi=200, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 13. Interpretation rules
#
# 1. Use cross-validation mean and standard deviation to discuss stability.
# 2. Compare augmentation and no-augmentation under identical folds.
# 3. Do not call the official test set an unseen-target test: it contains the
#    same six simulated satellite identities and elevations as development data.
# 4. Azimuth grouping prevents identical angles crossing fold boundaries, but
#    neighboring angles can remain close. A later blocked-angle or
#    leave-one-elevation-out experiment is still required for stronger claims.
# 5. A result close to 100% means this synthetic within-domain task is easy; it
#    does not establish transfer to measured ISAR data.

# %%
print("Study complete. Outputs:", OUT_DIR)
for path in sorted(OUT_DIR.glob("*.csv")):
    print(" -", path.name)
for path in sorted(OUT_DIR.glob("*.png")):
    print(" -", path.name)
