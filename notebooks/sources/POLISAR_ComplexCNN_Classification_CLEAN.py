# %% [markdown]
# # Pol-ISARSpaceTarget-1.0 Classification: Clean Complex CNN
#
# This notebook is intentionally classification-only.
#
# It implements exactly:
#
# 1. Load full-polarimetric complex ISAR channels using the dataset author's
#    `.dat` layout: `HH`, `HV`, `VH`, `VV`.
# 2. Build a Complex CNN.
# 3. Extract/tap a feature vector from a fully connected layer.
# 4. Remove per-sample normalization.
# 5. Compute real/imag mean and standard deviation from training data only.
# 6. Apply the same training statistics to train, validation, and test data.
# 7. Extract magnitude and phase features.
# 8. Concatenate complex-CNN FC features with magnitude-phase FC features.
# 9. Feed the fused vector to a classifier.
# 10. Compute classification metrics.
#
# This notebook contains only classification code and classification metrics.

# %%
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

try:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
except Exception:
    accuracy_score = confusion_matrix = f1_score = precision_score = recall_score = roc_auc_score = None
    StratifiedKFold = None


print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# %% [markdown]
# ## Configuration

# %%
@dataclass
class Config:
    data_root: str = "/content/drive/MyDrive/drdo/Pol_ISARSpaceTarget1.0"
    output_dir: str = "/content/drive/MyDrive/drdo/outputs/polisar_complex_classification_clean"
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 25
    learning_rate: float = 3e-4
    lr_candidates: Tuple[float, ...] = (1e-4, 3e-4, 1e-3)
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 8
    use_train_standardization: bool = True
    use_train_augmentation: bool = True
    augmentation_noise_std: float = 0.015
    kfold_splits: int = 5
    kfold_epochs: int = 8
    seed: int = 42
    amp: bool = False
    log_every_batches: int = 10


cfg = Config()
local_candidate = Path.cwd() / "Pol_ISARSpaceTarget1.0"
if not Path(cfg.data_root).exists() and local_candidate.exists():
    cfg.data_root = str(local_candidate)
Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
print(json.dumps(asdict(cfg), indent=2))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(cfg.seed)


# %% [markdown]
# ## Dataset Loader
#
# The model receives only the complex tensor. The filename is used only to create
# the class label.
#
# The reader below follows `Tools/DataReadPython.py` from the dataset:
# each `.dat` file stores interleaved float32 real and imaginary values, reshaped
# to `256 x 256`, and the four channels are placed as the `2 x 2` polarimetric
# matrix `[HH HV; VH VV]`.
#
# Important: there is no per-sample normalization. Mean/std statistics are
# computed from the training split only and then reused for validation and test.

# %%
POLS = ("HH", "HV", "VH", "VV")
SATELLITE_CLASSES = ("Capella", "HISEA1", "QPSSAR", "Radarsat2", "StarLink", "Worldview4")
CLASS_TO_ID = {name: idx for idx, name in enumerate(SATELLITE_CLASSES)}
DUPLICATE_RE = re.compile(r" \(\d+\)(?=\.[^.]+$)")


def is_drive_duplicate_name(name: str) -> bool:
    return bool(DUPLICATE_RE.search(name))


def infer_satellite_id(base: str) -> int:
    for class_name, class_id in CLASS_TO_ID.items():
        if base.startswith(class_name + "_"):
            return class_id
    raise ValueError(f"Could not infer class from filename base: {base}")


def read_full_pol_complex_official_layout(image_dir: Path, base: str) -> np.ndarray:
    data_size = (256, 256, 2, 2)
    complex_data = np.zeros(data_size, dtype=np.complex64)
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            file_path = image_dir / f"{base}_{POLS[idx]}.dat"
            combined_data = np.fromfile(file_path, dtype=np.float32)
            if combined_data.size != 256 * 256 * 2:
                raise ValueError(f"Unexpected file size for {file_path}: {combined_data.size} float32 values")
            real_part = combined_data[0::2].reshape(data_size[0:2])
            imag_part = combined_data[1::2].reshape(data_size[0:2])
            complex_data[:, :, i, j] = real_part + 1j * imag_part
    return complex_data


def polarimetric_matrix_to_channels(complex_data: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            complex_data[:, :, 0, 0],  # HH
            complex_data[:, :, 0, 1],  # HV
            complex_data[:, :, 1, 0],  # VH
            complex_data[:, :, 1, 1],  # VV
        ],
        axis=0,
    ).astype(np.complex64)


def resize_complex_tensor(x: torch.Tensor, image_size: int) -> torch.Tensor:
    if x.shape[-2:] == (image_size, image_size):
        return x
    real = F.interpolate(x.real.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)
    imag = F.interpolate(x.imag.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)
    return torch.complex(real.squeeze(0), imag.squeeze(0))


NormStats = Dict[str, torch.Tensor]


def standardize_complex_with_train_stats(x: torch.Tensor, stats: NormStats) -> torch.Tensor:
    real = (x.real - stats["real_mean"]) / stats["real_std"]
    imag = (x.imag - stats["imag_mean"]) / stats["imag_std"]
    return torch.complex(real, imag)


def augment_complex_tensor(x: torch.Tensor, noise_std: float = 0.015) -> torch.Tensor:
    """Light standard augmentation applied equally to real and imaginary parts."""
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=(-1,))
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=(-2,))
    k = int(torch.randint(0, 4, ()).item())
    if k:
        x = torch.rot90(x, k=k, dims=(-2, -1))
    if noise_std > 0:
        real_noise = torch.randn_like(x.real) * noise_std
        imag_noise = torch.randn_like(x.imag) * noise_std
        x = x + torch.complex(real_noise, imag_noise)
    return x


class PolISARClassificationDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str | Sequence[str],
        image_size: int,
        augment: bool = False,
        norm_stats: NormStats | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.augment = augment
        self.norm_stats = norm_stats
        self.split_names = [split] if isinstance(split, str) else list(split)
        self.samples: List[Tuple[Path, str]] = []

        for split_name in self.split_names:
            image_dir = self.data_root / "ImageSets" / "images" / split_name
            if not image_dir.exists():
                raise FileNotFoundError(f"Missing image directory: {image_dir}")

            bases = sorted(
                {
                    p.name.rsplit("_", 1)[0]
                    for p in image_dir.glob("*_HH.dat")
                    if not is_drive_duplicate_name(p.name)
                }
            )
            complete = [base for base in bases if all((image_dir / f"{base}_{pol}.dat").exists() for pol in POLS)]
            self.samples.extend((image_dir, base) for base in complete)
            expected = {"train": 1314, "valid": 162, "test": 162}.get(split_name)
            print(f"{split_name}: usable complete full-pol samples = {len(complete)}; expected = {expected}")

        print(
            f"combined {self.split_names}: samples = {len(self.samples)}; "
            f"augment = {self.augment}; train_stats_normalization = {self.norm_stats is not None}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_dir, base = self.samples[index]
        complex_matrix = read_full_pol_complex_official_layout(image_dir, base)
        x = torch.from_numpy(polarimetric_matrix_to_channels(complex_matrix))
        x = resize_complex_tensor(x, self.image_size)
        if self.norm_stats is not None:
            x = standardize_complex_with_train_stats(x, self.norm_stats)
        if self.augment:
            x = augment_complex_tensor(x, cfg.augmentation_noise_std)
        return {"x": x, "class_id": torch.tensor(infer_satellite_id(base), dtype=torch.long), "base": base}


def compute_complex_mean_std(dataset: PolISARClassificationDataset, indices: Sequence[int] | None = None, eps: float = 1e-6) -> NormStats:
    """Compute real/imag mean and std using only the training samples supplied."""
    use_indices = list(range(len(dataset))) if indices is None else list(indices)
    if not use_indices:
        raise ValueError("Cannot compute normalization statistics from an empty index list.")

    sum_real = torch.zeros(4, 1, 1)
    sum_imag = torch.zeros(4, 1, 1)
    sumsq_real = torch.zeros(4, 1, 1)
    sumsq_imag = torch.zeros(4, 1, 1)
    count = 0

    previous_stats = dataset.norm_stats
    previous_augment = dataset.augment
    dataset.norm_stats = None
    dataset.augment = False
    try:
        for index in use_indices:
            x = dataset[index]["x"]
            real = x.real.float()
            imag = x.imag.float()
            sum_real += real.sum(dim=(-2, -1), keepdim=True)
            sum_imag += imag.sum(dim=(-2, -1), keepdim=True)
            sumsq_real += (real * real).sum(dim=(-2, -1), keepdim=True)
            sumsq_imag += (imag * imag).sum(dim=(-2, -1), keepdim=True)
            count += real.shape[-2] * real.shape[-1]
    finally:
        dataset.norm_stats = previous_stats
        dataset.augment = previous_augment

    real_mean = sum_real / count
    imag_mean = sum_imag / count
    real_var = (sumsq_real / count - real_mean * real_mean).clamp_min(eps)
    imag_var = (sumsq_imag / count - imag_mean * imag_mean).clamp_min(eps)
    return {
        "real_mean": real_mean,
        "real_std": torch.sqrt(real_var),
        "imag_mean": imag_mean,
        "imag_std": torch.sqrt(imag_var),
    }


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "class_id": torch.stack([item["class_id"] for item in batch]),
        "base": [str(item["base"]) for item in batch],
    }


train_raw_ds = PolISARClassificationDataset(cfg.data_root, "train", cfg.image_size, augment=False, norm_stats=None)
train_norm_stats = compute_complex_mean_std(train_raw_ds) if cfg.use_train_standardization else None
if train_norm_stats is None:
    print("Normalization disabled: raw complex values are used.")
else:
    print("Normalization statistics computed from TRAIN split only:")
    print({key: value.flatten().tolist() for key, value in train_norm_stats.items()})

train_ds = PolISARClassificationDataset(cfg.data_root, "train", cfg.image_size, augment=cfg.use_train_augmentation, norm_stats=train_norm_stats)
valid_ds = PolISARClassificationDataset(cfg.data_root, "valid", cfg.image_size, augment=False, norm_stats=train_norm_stats)
test_ds = PolISARClassificationDataset(cfg.data_root, "test", cfg.image_size, augment=False, norm_stats=train_norm_stats)
print("Dataset sizes:", len(train_ds), len(valid_ds), len(test_ds))
sample = train_ds[0]
print("Sample:", sample["base"], sample["x"].shape, sample["x"].dtype, sample["class_id"])


def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


train_loader = make_loader(train_ds, shuffle=True)
valid_loader = make_loader(valid_ds, shuffle=False)
test_loader = make_loader(test_ds, shuffle=False)


# %% [markdown]
# ## Magnitude and Phase Features
#
# Magnitude and phase are explicitly separated. The old per-sample magnitude
# normalization has been removed. The complex input has already been standardized
# using training-set statistics only.

# %%
def magnitude_phase_tensor(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    mag = torch.abs(x)
    phase = torch.angle(x) / math.pi
    return mag.float(), phase.float()


mag_check, phase_check = magnitude_phase_tensor(sample["x"].unsqueeze(0))
print("Magnitude feature range:", float(mag_check.min()), float(mag_check.max()))
print("Phase feature range:", float(phase_check.min()), float(phase_check.max()))


# %% [markdown]
# ## Complex CNN Architecture
#
# This is the core architecture Mam asked to see.
#
# Complex convolution:
#
# `(a + jb)(c + jd) = (ac - bd) + j(ad + bc)`
#
# The FC feature vector is tapped from `self.fc_complex`.

# %%
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.real = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = self.real(x.real) - self.imag(x.imag)
        imag = self.real(x.imag) + self.imag(x.real)
        return torch.complex(real, imag)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.real_bn = nn.BatchNorm2d(channels)
        self.imag_bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.real_bn(x.real), self.imag_bn(x.imag))


class ComplexModReLU(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(x)
        activated = F.relu(mag + self.bias)
        phase = x / (mag + self.eps)
        return activated.to(x.dtype) * phase


class ComplexResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.act1 = ComplexModReLU(out_channels)
        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(out_channels)
        self.act2 = ComplexModReLU(out_channels)
        self.proj = None
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                ComplexConv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                ComplexBatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + identity)


class MagnitudePhaseFCBranch(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag_norm, phase_norm = magnitude_phase_tensor(x)
        return self.encoder(torch.cat([mag_norm, phase_norm], dim=1))


class ComplexCNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ComplexConv2d(4, 32, kernel_size=5, stride=2, padding=2),
            ComplexBatchNorm2d(32),
            ComplexModReLU(32),
        )
        self.stage1 = nn.Sequential(ComplexResidualBlock(32, 48, stride=2), ComplexResidualBlock(48, 48))
        self.stage2 = nn.Sequential(ComplexResidualBlock(48, 96, stride=2), ComplexResidualBlock(96, 96))
        self.stage3 = nn.Sequential(ComplexResidualBlock(96, 160, stride=2), ComplexResidualBlock(160, 160))

        # FC layer where complex CNN features are tapped.
        self.fc_complex = nn.Sequential(
            nn.Linear(320, 192),
            nn.GELU(),
            nn.Linear(192, 128),
            nn.Tanh(),
        )
        self.fc_mag_phase = MagnitudePhaseFCBranch(out_dim=128)

        # Concatenated vector = [complex_fc_features, magnitude_phase_features].
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.stem(x)
        z = self.stage1(z)
        z = self.stage2(z)
        z = self.stage3(z)

        real_imag_map = torch.cat([z.real, z.imag], dim=1)
        pooled = real_imag_map.mean(dim=(-2, -1))
        complex_fc_features = self.fc_complex(pooled)
        mag_phase_features = self.fc_mag_phase(x)
        fused_features = torch.cat([complex_fc_features, mag_phase_features], dim=1)
        logits = self.classifier(fused_features)
        return {
            "logits": logits,
            "complex_fc_features": complex_fc_features,
            "mag_phase_features": mag_phase_features,
            "fused_features": fused_features,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexCNNClassifier(num_classes=len(SATELLITE_CLASSES)).to(device)
print(model)
print("Trainable parameters:", f"{count_parameters(model):,}")

with torch.no_grad():
    smoke = sample["x"].unsqueeze(0).to(device)
    out = model(smoke)
    print("logits:", out["logits"].shape)
    print("complex_fc_features:", out["complex_fc_features"].shape, float(out["complex_fc_features"].min()), float(out["complex_fc_features"].max()))
    print("mag_phase_features:", out["mag_phase_features"].shape, float(out["mag_phase_features"].min()), float(out["mag_phase_features"].max()))
    print("fused_features:", out["fused_features"].shape)


# %% [markdown]
# ## Metrics

# %%
def classification_metrics(y_true: Sequence[int], logits: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    pred = probs.argmax(axis=1)
    y = np.array(y_true)
    metrics = {"accuracy": float((pred == y).mean())}
    if accuracy_score is not None:
        metrics.update(
            {
                "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
            }
        )
        try:
            metrics["auc_ovr"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
        except Exception:
            metrics["auc_ovr"] = float("nan")
    return metrics


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {"x": batch["x"].to(device), "class_id": batch["class_id"].to(device), "base": batch["base"]}


def run_epoch(model_obj: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, phase: str) -> Dict[str, float]:
    is_train = optimizer is not None
    model_obj.train(is_train)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    logits_all: List[torch.Tensor] = []
    targets: List[int] = []
    correct = 0
    seen = 0

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model_obj(batch["x"])
            loss = criterion(outputs["logits"], batch["class_id"])
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_obj.parameters(), cfg.grad_clip_norm)
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        preds = outputs["logits"].detach().argmax(dim=1)
        correct += int((preds == batch["class_id"]).sum().item())
        seen += int(batch["class_id"].numel())
        logits_all.append(outputs["logits"].detach().cpu())
        targets.extend(batch["class_id"].detach().cpu().tolist())

        if batch_idx == 1 or batch_idx == len(loader) or batch_idx % cfg.log_every_batches == 0:
            print(f"{phase} [{batch_idx:04d}/{len(loader):04d}] loss={total_loss / batch_idx:.4f} acc={correct / max(seen, 1):.3f}", flush=True)

    logits_cat = torch.cat(logits_all, dim=0)
    metrics = classification_metrics(targets, logits_cat)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


# %% [markdown]
# ## Verification Protocol: LR, Augmentation, K-Fold
#
# Accuracy of `1.0` should not be interpreted as proof of a perfect model.
# It can indicate an easy split, data leakage, or that the simulated target
# signatures are highly separable.
#
# The cells below add the checks Mam suggested:
#
# 1. learning-rate sweep
# 2. standard training augmentation
# 3. stratified K-fold cross-validation
#
# This is done before final model training. The original test split is not used
# inside LR selection or K-fold. K-fold is performed on `train + valid`; the
# official test split remains a final holdout and is evaluated only once after
# final settings are selected.

# %%
RUN_LR_SWEEP = False
RUN_KFOLD_STUDY = False
out_dir = Path(cfg.output_dir)


def train_small_protocol_run(
    train_loader_obj: DataLoader,
    valid_loader_obj: DataLoader,
    lr: float,
    epochs: int,
    tag: str,
) -> Dict[str, float]:
    seed_everything(cfg.seed)
    model_obj = ComplexCNNClassifier(num_classes=len(SATELLITE_CLASSES)).to(device)
    optimizer_obj = torch.optim.Adam(model_obj.parameters(), lr=lr)
    best_row: Dict[str, float] = {"valid_f1_macro": -1.0, "valid_accuracy": -1.0}

    for epoch_idx in range(epochs):
        train_metrics = run_epoch(model_obj, train_loader_obj, optimizer_obj, phase=f"{tag} epoch {epoch_idx + 1} train")
        valid_metrics = run_epoch(model_obj, valid_loader_obj, None, phase=f"{tag} epoch {epoch_idx + 1} valid")
        row = {
            "tag": tag,
            "epoch": float(epoch_idx + 1),
            "lr": float(lr),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        }
        current = row.get("valid_f1_macro", row["valid_accuracy"])
        if current > best_row.get("valid_f1_macro", -1.0):
            best_row = row
    return best_row


if RUN_LR_SWEEP:
    lr_rows = []
    for lr in cfg.lr_candidates:
        print(f"\nLR sweep running: lr={lr}")
        row = train_small_protocol_run(train_loader, valid_loader, lr=lr, epochs=cfg.kfold_epochs, tag=f"lr_{lr}")
        lr_rows.append(row)
        print(json.dumps(row, indent=2))
    (out_dir / "lr_sweep_results.json").write_text(json.dumps(lr_rows, indent=2))
    print("LR sweep results saved:", out_dir / "lr_sweep_results.json")
else:
    print("LR sweep skipped. Set RUN_LR_SWEEP = True to run it.")


if RUN_KFOLD_STUDY:
    if StratifiedKFold is None:
        raise RuntimeError("scikit-learn is required for StratifiedKFold.")

    combined_raw_ds = PolISARClassificationDataset(cfg.data_root, ["train", "valid"], cfg.image_size, augment=False, norm_stats=None)
    labels = [infer_satellite_id(base) for _, base in combined_raw_ds.samples]

    kfold = StratifiedKFold(n_splits=cfg.kfold_splits, shuffle=True, random_state=cfg.seed)
    fold_rows = []
    for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels), start=1):
        print(f"\nK-fold {fold_idx}/{cfg.kfold_splits}")
        fold_stats = compute_complex_mean_std(combined_raw_ds, train_idx.tolist()) if cfg.use_train_standardization else None
        combined_aug_ds = PolISARClassificationDataset(cfg.data_root, ["train", "valid"], cfg.image_size, augment=True, norm_stats=fold_stats)
        combined_clean_ds = PolISARClassificationDataset(cfg.data_root, ["train", "valid"], cfg.image_size, augment=False, norm_stats=fold_stats)
        fold_train_loader = make_loader(Subset(combined_aug_ds, train_idx.tolist()), shuffle=True)
        fold_valid_loader = make_loader(Subset(combined_clean_ds, valid_idx.tolist()), shuffle=False)
        row = train_small_protocol_run(
            fold_train_loader,
            fold_valid_loader,
            lr=cfg.learning_rate,
            epochs=cfg.kfold_epochs,
            tag=f"fold_{fold_idx}",
        )
        row["fold"] = float(fold_idx)
        fold_rows.append(row)
        print(json.dumps(row, indent=2))

    summary = {
        "folds": fold_rows,
        "mean_valid_accuracy": float(np.mean([row["valid_accuracy"] for row in fold_rows])),
        "std_valid_accuracy": float(np.std([row["valid_accuracy"] for row in fold_rows])),
        "mean_valid_f1_macro": float(np.mean([row["valid_f1_macro"] for row in fold_rows])),
        "std_valid_f1_macro": float(np.std([row["valid_f1_macro"] for row in fold_rows])),
    }
    (out_dir / "kfold_results.json").write_text(json.dumps(summary, indent=2))
    print("K-fold results saved:", out_dir / "kfold_results.json")
    print(json.dumps(summary, indent=2))
else:
    print("K-fold skipped. Set RUN_KFOLD_STUDY = True to run it.")


# %% [markdown]
# ## Select Final Settings
#
# If LR sweep is run, choose the LR with the best validation macro-F1.
# If it is skipped, use the default LR from `cfg.learning_rate`.
# Augmentation remains enabled for final training because the guide asked to
# check standard augmentation.

# %%
selected_lr = cfg.learning_rate
selected_augmentation = cfg.use_train_augmentation
lr_sweep_path = out_dir / "lr_sweep_results.json"
if lr_sweep_path.exists():
    lr_rows = json.loads(lr_sweep_path.read_text())
    if lr_rows:
        best_lr_row = max(lr_rows, key=lambda row: row.get("valid_f1_macro", row.get("valid_accuracy", -1.0)))
        selected_lr = float(best_lr_row["lr"])

print(
    json.dumps(
        {
            "selected_lr": selected_lr,
            "selected_augmentation": selected_augmentation,
            "selection_note": "Use LR sweep/K-fold results when available; otherwise use conservative default settings.",
        },
        indent=2,
    )
)


# %% [markdown]
# ## Final Training After Verification
#
# Final training uses `train + valid` with the selected settings. The official
# `test` split is untouched until the next section.

# %%
final_train_ds = PolISARClassificationDataset(
    cfg.data_root,
    ["train", "valid"],
    cfg.image_size,
    augment=selected_augmentation,
    norm_stats=train_norm_stats,
)
final_train_loader = make_loader(final_train_ds, shuffle=True)
final_model = ComplexCNNClassifier(num_classes=len(SATELLITE_CLASSES)).to(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=selected_lr)
final_history_rows: List[Dict[str, float]] = []
final_path = out_dir / "final_complex_cnn_classifier.pt"

for epoch in range(cfg.epochs):
    t0 = time.time()
    train_metrics = run_epoch(final_model, final_train_loader, final_optimizer, phase=f"final epoch {epoch + 1} train")
    row = {
        "epoch": float(epoch + 1),
        "lr": float(final_optimizer.param_groups[0]["lr"]),
        **{f"train_{key}": value for key, value in train_metrics.items()},
        "seconds": time.time() - t0,
    }
    final_history_rows.append(row)
    print(json.dumps(row, indent=2), flush=True)
    torch.save(
        {
            "model": final_model.state_dict(),
            "epoch": epoch,
            "classes": SATELLITE_CLASSES,
            "selected_lr": selected_lr,
            "selected_augmentation": selected_augmentation,
        },
        final_path,
    )

(out_dir / "final_training_history.json").write_text(json.dumps(final_history_rows, indent=2))
print("Final training history saved:", out_dir / "final_training_history.json")
print("Final checkpoint saved:", final_path)


# %% [markdown]
# ## Final Test Evaluation
#
# This is the first time the final model is evaluated on the official test
# split. Do not use the test result to tune LR, augmentation, or architecture.

# %%
checkpoint = torch.load(final_path, map_location=device)
final_model.load_state_dict(checkpoint["model"])
test_metrics = run_epoch(final_model, test_loader, None, phase="final test")
print("Final test metrics:")
print(json.dumps(test_metrics, indent=2))


@torch.no_grad()
def collect_predictions(model_obj: nn.Module, loader: DataLoader, metrics: Dict[str, float]) -> Dict[str, object]:
    model_obj.eval()
    targets: List[int] = []
    preds: List[int] = []
    bases: List[str] = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model_obj(batch["x"])["logits"]
        targets.extend(batch["class_id"].detach().cpu().tolist())
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        bases.extend(batch["base"])
    return {"targets": targets, "predictions": preds, "bases": bases, "classes": list(SATELLITE_CLASSES), "metrics": metrics}


predictions = collect_predictions(final_model, test_loader, test_metrics)
(out_dir / "final_test_predictions.json").write_text(json.dumps(predictions, indent=2))
if confusion_matrix is not None:
    cm = confusion_matrix(predictions["targets"], predictions["predictions"])
    np.save(out_dir / "final_test_confusion_matrix.npy", cm)
    print("Final confusion matrix rows=true cols=pred:")
    print(cm)


# %% [markdown]
# ## Why 100% Can Be Suspicious
#
# If magnitude-only gives 100%, phase and feature selection are not needed to
# improve this official split. They may still be used for analysis or robustness,
# but the correct scientific response is to audit the split and create a harder
# validation protocol.
#
# Suggested next checks:
#
# - verify no filename/label leakage into the model input
# - verify test loader uses `test_ds`
# - verify no base-name overlap across train/valid/test
# - run the LR sweep and K-fold sections above
# - compare training with and without augmentation
# - try leave-one-elevation-out validation
# - try random class-balanced repeated splits
# - try noise/occlusion/low-SNR robustness tests
