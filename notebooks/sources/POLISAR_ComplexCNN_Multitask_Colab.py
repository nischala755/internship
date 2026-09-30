# %% [markdown]
# # Robust Polarimetric ISAR Classification via Complex Feature Fusion
#
# Research notebook for `Pol-ISARSpaceTarget-1.0`.
#
# Primary task: six-class satellite recognition from full-polarimetric complex ISAR.
#
# Auxiliary task: rotated component detection from the provided labels.
#
# This notebook is designed to run in Google Colab or a VS Code notebook connected
# to a Colab GPU runtime.

# %%
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except Exception:
    accuracy_score = None
    confusion_matrix = None
    f1_score = None
    precision_score = None
    recall_score = None
    roc_auc_score = None


print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    try:
        os.system("nvidia-smi")
    except Exception as exc:
        print("nvidia-smi unavailable:", exc)


# %% [markdown]
# ## Configuration
#
# Set `DATA_ROOT` to the extracted dataset folder that contains:
#
# - `ImageSets/images/train`
# - `ImageSets/labelTxt/train`
# - `Annotations`
#
# For Colab Drive, this is usually something like:
#
# `/content/drive/MyDrive/drdo/Pol_ISARSpaceTarget1.0`

# %%
@dataclass
class Config:
    data_root: str = "/content/drive/MyDrive/drdo/Pol_ISARSpaceTarget1.0"
    output_dir: str = "/content/drive/MyDrive/drdo/outputs/polisar_complex_multitask"
    image_size: int = 128
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    seed: int = 42
    amp: bool = False
    detection_weight: float = 0.30
    objectness_weight: float = 1.0
    det_class_weight: float = 1.0
    box_weight: float = 4.0
    angle_weight: float = 1.0
    early_stop_patience: int = 12
    resume_checkpoint: str = ""
    representation: str = "complex_fusion"
    log_every_batches: int = 10

    @property
    def stride(self) -> int:
        return 16

    @property
    def grid_size(self) -> int:
        return self.image_size // self.stride


cfg = Config()

# If running locally from the extracted workspace, this fallback may work.
local_candidate = Path.cwd() / "Pol_ISARSpaceTarget1.0"
if not Path(cfg.data_root).exists() and local_candidate.exists():
    cfg.data_root = str(local_candidate)

Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
print(json.dumps(asdict(cfg), indent=2))


# %% [markdown]
# ## Dataset and Preprocessing
#
# Each training sample groups four `.dat` files:
#
# `HH`, `HV`, `VH`, `VV`
#
# The raw `.dat` format is interleaved float32 real/imaginary values for a
# `256 x 256` complex image.

# %%
SATELLITE_CLASSES = ["Capella", "HISEA1", "QPSSAR", "Radarsat2", "StarLink", "Worldview4"]
COMPONENT_CLASSES = [
    "cuboid_body",
    "optical_lens",
    "panel_antenna",
    "parabolic_antenna",
    "plate_body",
    "plate_fitting",
    "polygon_body",
    "rod_fitting",
    "solar_array",
]
POLS = ["HH", "HV", "VH", "VV"]
SAT_TO_ID = {name: idx for idx, name in enumerate(SATELLITE_CLASSES)}
COMP_TO_ID = {name: idx for idx, name in enumerate(COMPONENT_CLASSES)}
DRIVE_DUPLICATE_RE = re.compile(r" \(\d+\)(?=\.[^.]+$)")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def infer_satellite_id(base_name: str) -> int:
    for sat, idx in SAT_TO_ID.items():
        if base_name.startswith(sat + "_"):
            return idx
    raise ValueError(f"Cannot infer satellite class from {base_name}")


def is_drive_duplicate_name(name: str) -> bool:
    return DRIVE_DUPLICATE_RE.search(name) is not None


def dataset_upload_report(data_root: str) -> None:
    root = Path(data_root)
    checks = [
        ("train dat", root / "ImageSets" / "images" / "train", "*.dat", 5256),
        ("valid dat", root / "ImageSets" / "images" / "valid", "*.dat", 648),
        ("test dat", root / "ImageSets" / "images" / "test", "*.dat", 648),
        ("train labels", root / "ImageSets" / "labelTxt" / "train", "*.txt", 1314),
        ("valid labels", root / "ImageSets" / "labelTxt" / "valid", "*.txt", 162),
        ("test labels", root / "ImageSets" / "labelTxt" / "test", "*.txt", 162),
    ]
    print("Dataset root:", root)
    for name, path, pattern, expected in checks:
        files = list(path.glob(pattern)) if path.exists() else []
        duplicate_files = [p for p in files if is_drive_duplicate_name(p.name)]
        clean_files = [p for p in files if not is_drive_duplicate_name(p.name)]
        status = "OK" if len(clean_files) == expected else "CHECK"
        print(
            f"{name:13s} total={len(files):5d} clean={len(clean_files):5d} "
            f"duplicates={len(duplicate_files):3d} expected={expected:5d} {status}"
        )
        if duplicate_files[:3]:
            print("  duplicate examples:", [p.name for p in duplicate_files[:3]])


def read_complex_dat(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size != 256 * 256 * 2:
        raise ValueError(f"Unexpected file size for {path}: {data.size} float32 values")
    complex_image = data[0::2] + 1j * data[1::2]
    return complex_image.reshape(256, 256).astype(np.complex64)


def polygon_to_box(points: np.ndarray) -> Tuple[float, float, float, float, float]:
    center = points.mean(axis=0)
    edge01 = points[1] - points[0]
    edge12 = points[2] - points[1]
    width = float(np.linalg.norm(edge01))
    height = float(np.linalg.norm(edge12))
    angle = float(math.atan2(edge01[1], edge01[0]))
    return float(center[0]), float(center[1]), width, height, angle


def parse_label_file(path: Path) -> List[Dict[str, float]]:
    objects: List[Dict[str, float]] = []
    if not path.exists():
        return objects
    for raw in path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 10:
            continue
        coords = np.array([float(x) for x in parts[:8]], dtype=np.float32).reshape(4, 2)
        class_name = parts[8]
        if class_name not in COMP_TO_ID:
            continue
        cx, cy, w, h, angle = polygon_to_box(coords)
        objects.append(
            {
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "angle": angle,
                "class_id": float(COMP_TO_ID[class_name]),
            }
        )
    return objects


def resize_complex_tensor(x: torch.Tensor, image_size: int) -> torch.Tensor:
    if x.shape[-1] == image_size and x.shape[-2] == image_size:
        return x
    real = F.interpolate(x.real.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)
    imag = F.interpolate(x.imag.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)
    return torch.complex(real.squeeze(0), imag.squeeze(0))


def normalize_complex_per_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale each complex polarization so real/imag values stay near [-1, 1]."""
    max_amp = torch.abs(x).amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return x / max_amp.to(x.dtype)


def magnitude_phase_tensor(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return [mag, phase] features for HH/HV/VH/VV normalized to [-1, 1]."""
    mag = torch.abs(x)
    mag_max = mag.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    mag_norm = 2.0 * (mag / mag_max) - 1.0
    phase_norm = torch.angle(x) / math.pi
    return torch.cat([mag_norm, phase_norm], dim=1).float()


def build_detection_target(
    objects: Sequence[Dict[str, float]], image_size: int, grid_size: int, num_classes: int
) -> torch.Tensor:
    target = torch.zeros(grid_size, grid_size, 1 + num_classes + 6, dtype=torch.float32)
    scale = image_size / 256.0
    cell = image_size / grid_size

    for obj in objects:
        cx = float(obj["cx"]) * scale
        cy = float(obj["cy"]) * scale
        w = max(float(obj["w"]) * scale, 1.0)
        h = max(float(obj["h"]) * scale, 1.0)
        angle = float(obj["angle"])
        gx = min(max(int(cx / cell), 0), grid_size - 1)
        gy = min(max(int(cy / cell), 0), grid_size - 1)
        existing_area = target[gy, gx, 1 + num_classes + 2] * target[gy, gx, 1 + num_classes + 3]
        area = (w / image_size) * (h / image_size)
        if target[gy, gx, 0] > 0 and existing_area >= area:
            continue

        target[gy, gx, 0] = 1.0
        class_id = int(obj["class_id"])
        target[gy, gx, 1 + class_id] = 1.0
        offset_x = (cx / cell) - gx
        offset_y = (cy / cell) - gy
        box_start = 1 + num_classes
        target[gy, gx, box_start : box_start + 6] = torch.tensor(
            [
                offset_x,
                offset_y,
                w / image_size,
                h / image_size,
                math.sin(angle),
                math.cos(angle),
            ],
            dtype=torch.float32,
        )

    return target


class PolISARDataset(Dataset):
    def __init__(self, data_root: str, split: str, image_size: int, grid_size: int) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.grid_size = grid_size
        self.image_dir = self.data_root / "ImageSets" / "images" / split
        self.label_dir = self.data_root / "ImageSets" / "labelTxt" / split
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image split directory: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Missing label split directory: {self.label_dir}")

        duplicate_dat = [p.name for p in self.image_dir.glob("*.dat") if is_drive_duplicate_name(p.name)]
        duplicate_txt = [p.name for p in self.label_dir.glob("*.txt") if is_drive_duplicate_name(p.name)]
        if duplicate_dat or duplicate_txt:
            print(
                f"[{split}] Ignoring Drive duplicate files: "
                f"{len(duplicate_dat)} dat, {len(duplicate_txt)} labels"
            )

        bases = sorted(
            {
                p.name.rsplit("_", 1)[0]
                for p in self.image_dir.glob("*_HH.dat")
                if not is_drive_duplicate_name(p.name)
            }
        )
        self.samples = [base for base in bases if all((self.image_dir / f"{base}_{pol}.dat").exists() for pol in POLS)]
        if not self.samples:
            raise RuntimeError(f"No complete HH/HV/VH/VV samples found in {self.image_dir}")
        expected = {"train": 1314, "valid": 162, "test": 162}.get(split)
        if expected is not None and len(self.samples) != expected:
            print(f"[{split}] Warning: usable complete samples={len(self.samples)}, expected={expected}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        base = self.samples[index]
        channels = []
        for pol in POLS:
            arr = read_complex_dat(self.image_dir / f"{base}_{pol}.dat")
            channels.append(torch.from_numpy(arr))
        x = torch.stack(channels, dim=0)
        x = resize_complex_tensor(x, self.image_size)
        x = normalize_complex_per_channel(x)

        label_path = self.label_dir / f"{base}.txt"
        objects = parse_label_file(label_path)
        det_target = build_detection_target(objects, self.image_size, self.grid_size, len(COMPONENT_CLASSES))
        return {
            "x": x,
            "class_id": torch.tensor(infer_satellite_id(base), dtype=torch.long),
            "det_target": det_target,
            "base": base,
        }


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "class_id": torch.stack([item["class_id"] for item in batch]),
        "det_target": torch.stack([item["det_target"] for item in batch]),
        "base": [str(item["base"]) for item in batch],
    }


seed_everything(cfg.seed)
dataset_upload_report(cfg.data_root)
train_ds = PolISARDataset(cfg.data_root, "train", cfg.image_size, cfg.grid_size)
valid_ds = PolISARDataset(cfg.data_root, "valid", cfg.image_size, cfg.grid_size)
test_ds = PolISARDataset(cfg.data_root, "test", cfg.image_size, cfg.grid_size)
print("Dataset sizes:", len(train_ds), len(valid_ds), len(test_ds))
sample = train_ds[0]
print("Sample:", sample["base"], sample["x"].shape, sample["x"].dtype, sample["det_target"].shape)
sample_mp = magnitude_phase_tensor(sample["x"].unsqueeze(0))
print(
    "Magnitude-phase feature range:",
    float(sample_mp.min()),
    float(sample_mp.max()),
    "shape:",
    tuple(sample_mp.shape),
)


# %% [markdown]
# ## Complex CNN Layers
#
# Complex convolution is implemented from the mathematical product:
#
# `(a + jb)(c + jd) = (ac - bd) + j(ad + bc)`
#
# BatchNorm is applied separately to real and imaginary components. This is a
# computationally efficient approximation; a full covariance complex BN is a
# possible improvement for a later ablation.

# %%
class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

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
        magnitude = torch.abs(x)
        activated = F.relu(magnitude + self.bias)
        phase = x / (magnitude + self.eps)
        return activated.to(x.dtype) * phase


class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.pool(x.real), self.pool(x.imag))


class ComplexResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.act1 = ComplexModReLU(out_channels)
        self.conv2 = ComplexConv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = ComplexBatchNorm2d(out_channels)
        self.act2 = ComplexModReLU(out_channels)
        self.proj: Optional[nn.Module]
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                ComplexConv2d(in_channels, out_channels, 1, stride=stride),
                ComplexBatchNorm2d(out_channels),
            )
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + identity)


class ComplexSEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(x).mean(dim=(-2, -1))
        weights = torch.sigmoid(self.fc2(F.relu(self.fc1(mag)))).unsqueeze(-1).unsqueeze(-1)
        self.last_weights = weights.detach().cpu()
        return x * weights.to(x.dtype)


class ComplexSpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(x)
        avg_map = mag.mean(dim=1, keepdim=True)
        max_map = mag.max(dim=1, keepdim=True).values
        attn = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        self.last_map = attn.detach().cpu()
        return x * attn.to(x.dtype)


class ComplexFusionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_attn = ComplexSEAttention(channels)
        self.spatial_attn = ComplexSpatialAttention()
        self.refine = ComplexResidualBlock(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attn(x)
        out = self.spatial_attn(out)
        return self.refine(out + x)


class MagnitudePhaseBranch(nn.Module):
    """Real-valued branch that explicitly taps normalized magnitude and phase."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mp = magnitude_phase_tensor(x)
        return self.proj(self.encoder(mp))


# %% [markdown]
# ## Multi-Task Complex CNN
#
# The shared backbone learns phase-preserving polarimetric features.
#
# Guide-required classification path:
#
# 1. Use a Complex CNN over the full HH/HV/VH/VV complex tensor.
# 2. Tap explicit magnitude and phase features from the same tensor.
# 3. Normalize magnitude and phase features to `[-1, 1]`.
# 4. Project complex and magnitude-phase streams to bounded feature vectors.
# 5. Concatenate both vectors and feed them to the classification head.
# 6. Compute classification metrics on train/validation/test splits.
#
# The classification head is the primary branch. The detection head is auxiliary
# and encourages component-aware features.

# %%
class ComplexPolISARNet(nn.Module):
    def __init__(self, num_sat_classes: int, num_component_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ComplexConv2d(4, 32, kernel_size=5, stride=2, padding=2),
            ComplexBatchNorm2d(32),
            ComplexModReLU(32),
        )
        self.stage1 = nn.Sequential(
            ComplexResidualBlock(32, 48, stride=2),
            ComplexResidualBlock(48, 48),
        )
        self.stage2 = nn.Sequential(
            ComplexResidualBlock(48, 96, stride=2),
            ComplexResidualBlock(96, 96),
        )
        self.stage3 = nn.Sequential(
            ComplexResidualBlock(96, 160, stride=2),
            ComplexResidualBlock(160, 160),
        )
        self.fusion = ComplexFusionBlock(160)
        self.complex_vector = nn.Sequential(
            nn.Linear(320, 192),
            nn.GELU(),
            nn.Linear(192, 128),
            nn.Tanh(),
        )
        self.mag_phase_branch = MagnitudePhaseBranch(out_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(256, 192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, num_sat_classes),
        )
        det_channels = 1 + num_component_classes + 6
        self.det_head = nn.Sequential(
            nn.Conv2d(320, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Conv2d(192, det_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mag_phase_vector = self.mag_phase_branch(x)
        z = self.stem(x)
        z = self.stage1(z)
        z = self.stage2(z)
        z = self.stage3(z)
        z = self.fusion(z)
        real_features = torch.cat([z.real, z.imag], dim=1)
        complex_pooled = real_features.mean(dim=(-2, -1))
        complex_vector = self.complex_vector(complex_pooled)
        fused_vector = torch.cat([complex_vector, mag_phase_vector], dim=1)
        class_logits = self.classifier(fused_vector)
        det_logits = self.det_head(real_features).permute(0, 2, 3, 1).contiguous()
        return {
            "class_logits": class_logits,
            "det_logits": det_logits,
            "complex_vector": complex_vector,
            "mag_phase_vector": mag_phase_vector,
            "fused_vector": fused_vector,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)
if device.type == "cuda":
    print("Using GPU:", torch.cuda.get_device_name(0))
model = ComplexPolISARNet(len(SATELLITE_CLASSES), len(COMPONENT_CLASSES)).to(device)
print(model)
print("Trainable parameters:", f"{count_parameters(model):,}")

with torch.no_grad():
    if "sample" in globals():
        smoke_x = sample["x"].unsqueeze(0).to(device)
    else:
        smoke_real = torch.randn(1, 4, cfg.image_size, cfg.image_size, device=device)
        smoke_imag = torch.randn(1, 4, cfg.image_size, cfg.image_size, device=device)
        smoke_x = torch.complex(smoke_real, smoke_imag)
    smoke_out = model(smoke_x)
    print("Smoke class logits:", tuple(smoke_out["class_logits"].shape))
    print("Smoke det logits:", tuple(smoke_out["det_logits"].shape))
    print("Complex vector range:", float(smoke_out["complex_vector"].min()), float(smoke_out["complex_vector"].max()))
    print("Mag-phase vector range:", float(smoke_out["mag_phase_vector"].min()), float(smoke_out["mag_phase_vector"].max()))
    print("Fused vector:", tuple(smoke_out["fused_vector"].shape))


# %% [markdown]
# ## Losses and Metrics

# %%
class MultiTaskLoss(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.cfg = config
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.obj_loss = nn.BCEWithLogitsLoss()
        self.det_cls_loss = nn.CrossEntropyLoss()
        self.box_loss = nn.SmoothL1Loss()

    def forward(
        self, outputs: Dict[str, torch.Tensor], class_target: torch.Tensor, det_target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        class_loss = self.cls_loss(outputs["class_logits"], class_target)
        det_logits = outputs["det_logits"]
        obj_target = det_target[..., 0]
        obj_loss = self.obj_loss(det_logits[..., 0], obj_target)

        positive = obj_target > 0.5
        if positive.any():
            det_class_logits = det_logits[..., 1 : 1 + len(COMPONENT_CLASSES)][positive]
            det_class_target = det_target[..., 1 : 1 + len(COMPONENT_CLASSES)][positive].argmax(dim=-1)
            det_class_loss = self.det_cls_loss(det_class_logits, det_class_target)

            box_start = 1 + len(COMPONENT_CLASSES)
            pred_box = det_logits[..., box_start : box_start + 6][positive]
            target_box = det_target[..., box_start : box_start + 6][positive]
            pred_box = pred_box.clone()
            pred_box[:, 0:2] = torch.sigmoid(pred_box[:, 0:2])
            pred_box[:, 2:4] = torch.sigmoid(pred_box[:, 2:4])
            pred_angle = F.normalize(pred_box[:, 4:6], dim=-1)
            target_angle = F.normalize(target_box[:, 4:6], dim=-1)
            box_loss = self.box_loss(pred_box[:, :4], target_box[:, :4])
            angle_loss = self.box_loss(pred_angle, target_angle)
        else:
            det_class_loss = det_logits.sum() * 0.0
            box_loss = det_logits.sum() * 0.0
            angle_loss = det_logits.sum() * 0.0

        detection_loss = (
            self.cfg.objectness_weight * obj_loss
            + self.cfg.det_class_weight * det_class_loss
            + self.cfg.box_weight * box_loss
            + self.cfg.angle_weight * angle_loss
        )
        total = class_loss + self.cfg.detection_weight * detection_loss
        return {
            "loss": total,
            "class_loss": class_loss.detach(),
            "detection_loss": detection_loss.detach(),
            "objectness_loss": obj_loss.detach(),
            "det_class_loss": det_class_loss.detach(),
            "box_loss": box_loss.detach(),
            "angle_loss": angle_loss.detach(),
        }


@torch.no_grad()
def classification_metrics(y_true: Sequence[int], logits: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred = probs.argmax(axis=1)
    y_true_np = np.array(y_true)
    metrics = {
        "accuracy": float((y_pred == y_true_np).mean()),
    }
    if accuracy_score is not None:
        metrics.update(
            {
                "precision_macro": float(precision_score(y_true_np, y_pred, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_true_np, y_pred, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y_true_np, y_pred, average="macro", zero_division=0)),
            }
        )
        try:
            metrics["auc_ovr"] = float(roc_auc_score(y_true_np, probs, multi_class="ovr"))
        except Exception:
            metrics["auc_ovr"] = float("nan")
    return metrics


# %% [markdown]
# ## Training and Evaluation

# %%
def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
        persistent_workers=False,
    )


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "x": batch["x"].to(device, non_blocking=True),
        "class_id": batch["class_id"].to(device, non_blocking=True),
        "det_target": batch["det_target"].to(device, non_blocking=True),
        "base": batch["base"],
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": asdict(cfg),
            "satellite_classes": SATELLITE_CLASSES,
            "component_classes": COMPONENT_CLASSES,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("best_metric", 0.0))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    phase: str,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: Dict[str, float] = {}
    n_batches = 0
    all_targets: List[int] = []
    all_logits: List[torch.Tensor] = []
    correct = 0
    seen = 0

    iterator: Iterable[Dict[str, torch.Tensor]] = loader
    total_batches = len(loader)
    print(f"{phase}: starting {total_batches} batches", flush=True)

    for batch_idx, batch in enumerate(iterator, start=1):
        batch = move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=cfg.amp and torch.cuda.is_available()):
                outputs = model(batch["x"])
                losses = criterion(outputs, batch["class_id"], batch["det_target"])

            if is_train:
                assert scaler is not None
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        n_batches += 1
        preds = outputs["class_logits"].detach().argmax(dim=1)
        correct += int((preds == batch["class_id"]).sum().item())
        seen += int(batch["class_id"].numel())
        all_targets.extend(batch["class_id"].detach().cpu().tolist())
        all_logits.append(outputs["class_logits"].detach().cpu())

        should_log = (
            batch_idx == 1
            or batch_idx == total_batches
            or batch_idx % max(cfg.log_every_batches, 1) == 0
        )
        if should_log:
            print(
                f"{phase} [{batch_idx:04d}/{total_batches:04d}] "
                f"loss={totals['loss'] / n_batches:.4f} "
                f"cls={totals['class_loss'] / n_batches:.4f} "
                f"det={totals['detection_loss'] / n_batches:.4f} "
                f"acc={correct / max(seen, 1):.3f}",
                flush=True,
            )

    avg = {key: value / max(n_batches, 1) for key, value in totals.items()}
    logits = torch.cat(all_logits, dim=0)
    avg.update(classification_metrics(all_targets, logits))
    return avg


train_loader = make_loader(train_ds, shuffle=True)
valid_loader = make_loader(valid_ds, shuffle=False)
test_loader = make_loader(test_ds, shuffle=False)

criterion = MultiTaskLoss(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and torch.cuda.is_available())

start_epoch = 0
best_valid_f1 = 0.0
if cfg.resume_checkpoint:
    start_epoch, best_valid_f1 = load_checkpoint(cfg.resume_checkpoint, model, optimizer, scheduler)
    print("Resumed from", cfg.resume_checkpoint, "start_epoch", start_epoch, "best_valid_f1", best_valid_f1)

history: List[Dict[str, float]] = []
best_path = Path(cfg.output_dir) / "best_complex_polisar_multitask.pt"
last_path = Path(cfg.output_dir) / "last_complex_polisar_multitask.pt"
epochs_without_improvement = 0
print("Checkpoints:")
print("  best:", best_path)
print("  last:", last_path)
print("Early stopping monitors validation macro-F1, fallback validation accuracy.")

for epoch in range(start_epoch, cfg.epochs):
    t0 = time.time()
    train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, phase=f"epoch {epoch + 1} train")
    valid_metrics = run_epoch(model, valid_loader, criterion, None, None, device, phase=f"epoch {epoch + 1} valid")
    scheduler.step()

    row = {
        "epoch": float(epoch),
        "lr": float(scheduler.get_last_lr()[0]),
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"valid_{k}": v for k, v in valid_metrics.items()},
        "seconds": time.time() - t0,
    }
    history.append(row)
    print(json.dumps(row, indent=2))

    current = valid_metrics.get("f1_macro", valid_metrics["accuracy"])
    save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_valid_f1)
    if current > best_valid_f1:
        best_valid_f1 = current
        epochs_without_improvement = 0
        save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_valid_f1)
        print(f"Saved best checkpoint: {best_path} with monitored score={best_valid_f1:.4f}")
    else:
        epochs_without_improvement += 1
        print(
            f"No validation improvement for {epochs_without_improvement}/"
            f"{cfg.early_stop_patience} epochs."
        )

    if epochs_without_improvement >= cfg.early_stop_patience:
        print("Early stopping triggered.")
        break

history_path = Path(cfg.output_dir) / "history.json"
history_path.write_text(json.dumps(history, indent=2))
print("Training history saved:", history_path)


# %% [markdown]
# ## Final Test Evaluation

# %%
if best_path.exists():
    load_checkpoint(str(best_path), model)
    print("Loaded best checkpoint for test evaluation.")

test_metrics = run_epoch(model, test_loader, criterion, None, None, device, phase="test")
print("Test metrics:")
print(json.dumps(test_metrics, indent=2))

all_targets: List[int] = []
all_preds: List[int] = []
all_probs: List[np.ndarray] = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = move_batch(batch, device)
        outputs = model(batch["x"])
        probs = torch.softmax(outputs["class_logits"], dim=1).cpu().numpy()
        all_probs.extend(list(probs))
        all_preds.extend(probs.argmax(axis=1).tolist())
        all_targets.extend(batch["class_id"].cpu().tolist())

if confusion_matrix is not None:
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion matrix rows=true cols=pred:")
    print(cm)
    np.save(Path(cfg.output_dir) / "test_confusion_matrix.npy", cm)

with open(Path(cfg.output_dir) / "test_predictions.json", "w") as f:
    json.dump(
        {
            "targets": all_targets,
            "predictions": all_preds,
            "classes": SATELLITE_CLASSES,
            "metrics": test_metrics,
        },
        f,
        indent=2,
    )

print("Outputs written to:", cfg.output_dir)


# %% [markdown]
# ## Training Curves and Result Plots
#
# This cell creates the figures needed for discussion with the guide:
#
# - learning rate curve
# - total loss curve
# - classification loss curve
# - detection loss curve
# - accuracy curve
# - macro-F1 curve
# - confusion matrix

# %%
import matplotlib.pyplot as plt


def load_history(history_file: Path) -> List[Dict[str, float]]:
    if not history_file.exists():
        raise FileNotFoundError(f"Missing history file: {history_file}")
    return json.loads(history_file.read_text())


def plot_metric(history_rows: List[Dict[str, float]], keys: Sequence[str], title: str, ylabel: str, out_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history_rows]
    plt.figure(figsize=(8, 5))
    for key in keys:
        values = [row.get(key, float("nan")) for row in history_rows]
        plt.plot(epochs, values, marker="o", linewidth=2, label=key)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()


def plot_confusion_matrix(cm_path: Path, out_path: Path) -> None:
    if not cm_path.exists():
        print("Confusion matrix file not found:", cm_path)
        return
    cm = np.load(cm_path)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(range(len(SATELLITE_CLASSES)), SATELLITE_CLASSES, rotation=45, ha="right")
    plt.yticks(range(len(SATELLITE_CLASSES)), SATELLITE_CLASSES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()


plots_dir = Path(cfg.output_dir) / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
history_rows = load_history(Path(cfg.output_dir) / "history.json")

plot_metric(history_rows, ["lr"], "Learning Rate Schedule", "Learning Rate", plots_dir / "learning_rate.png")
plot_metric(history_rows, ["train_loss", "valid_loss"], "Total Loss", "Loss", plots_dir / "total_loss.png")
plot_metric(
    history_rows,
    ["train_class_loss", "valid_class_loss"],
    "Classification Loss",
    "Loss",
    plots_dir / "classification_loss.png",
)
plot_metric(
    history_rows,
    ["train_detection_loss", "valid_detection_loss"],
    "Detection Loss",
    "Loss",
    plots_dir / "detection_loss.png",
)
plot_metric(
    history_rows,
    ["train_accuracy", "valid_accuracy"],
    "Classification Accuracy",
    "Accuracy",
    plots_dir / "accuracy.png",
)
plot_metric(
    history_rows,
    ["train_f1_macro", "valid_f1_macro"],
    "Macro F1 Score",
    "Macro F1",
    plots_dir / "macro_f1.png",
)
plot_confusion_matrix(Path(cfg.output_dir) / "test_confusion_matrix.npy", plots_dir / "test_confusion_matrix.png")

print("Saved plots to:", plots_dir)


# %% [markdown]
# ## README: Ablation Experiments for Guide Review
#
# The first full model result is strong, so the next scientifically important
# step is an ablation study. This section compares feature representations under
# the same official train/valid/test split.
#
# Dataset clarification:
#
# - The dataset has more than 20k files total.
# - The actual full-polarimetric classification samples are `1,638`.
# - Each sample is made of four complex `.dat` files: `HH`, `HV`, `VH`, `VV`.
# - Official split:
#   - train: `1,314` samples
#   - valid: `162` samples
#   - test: `162` samples
#
# Ablations implemented here:
#
# | Experiment | Input Features | Purpose |
# |---|---|---|
# | Magnitude only | `|HH|, |HV|, |VH|, |VV|` | Tests scattering-strength information only |
# | Phase only | `angle(HH), angle(HV), angle(VH), angle(VV)` | Tests phase-only discriminability |
# | Magnitude + Phase | magnitude plus phase for all polarizations | Tests guide-requested feature fusion |
# | Real + Imaginary | real and imaginary parts for all polarizations | Tests raw complex information in real-valued form |
# | Complex only | complex CNN branch only | Tests the guide-required complex CNN without explicit mag-phase fusion |
# | Final model | complex CNN plus magnitude-phase fusion plus auxiliary detection | Main proposed model |
#
# The ablation models below are classification-only so that each representation
# can be compared fairly. The final model result with auxiliary detection is
# already saved above.
#
# Outputs are saved to:
#
# `/content/drive/MyDrive/drdo/outputs/polisar_complex_multitask/ablations/`
#
# Recommended workflow:
#
# 1. Keep `image_size=128` for quick ablations.
# 2. Run `ablation_epochs=10` first.
# 3. Increase to `20` only if the guide asks for stronger ablation runs.
# 4. Report test accuracy, macro-F1, and AUC for each feature setting.

# %%
def ablation_features(x: torch.Tensor, mode: str, eps: float = 1e-6) -> torch.Tensor:
    """Convert complex polarimetric input into real-valued ablation features."""
    if mode == "magnitude":
        mag = torch.abs(x)
        mag_max = mag.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return (2.0 * (mag / mag_max) - 1.0).float()
    if mode == "phase":
        return (torch.angle(x) / math.pi).float()
    if mode == "mag_phase":
        return magnitude_phase_tensor(x)
    if mode == "real_imag":
        real = x.real
        imag = x.imag
        scale = torch.maximum(real.abs(), imag.abs()).amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return torch.cat([real / scale, imag / scale], dim=1).float()
    raise ValueError(f"Unknown ablation mode: {mode}")


class RealFeatureClassifier(nn.Module):
    """Compact CNN for real-valued ablation feature tensors."""

    def __init__(self, in_channels: int, num_classes: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComplexOnlyClassifier(nn.Module):
    """Complex CNN baseline without explicit magnitude-phase branch or detection."""

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
        self.fusion = ComplexFusionBlock(160)
        self.classifier = nn.Sequential(
            nn.Linear(320, 192),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.stage1(z)
        z = self.stage2(z)
        z = self.stage3(z)
        z = self.fusion(z)
        features = torch.cat([z.real, z.imag], dim=1).mean(dim=(-2, -1))
        return self.classifier(features)


@torch.no_grad()
def evaluate_classifier_ablation(model: nn.Module, loader: DataLoader, mode: str, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    targets: List[int] = []
    logits_list: List[torch.Tensor] = []
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.05)
    for batch in loader:
        batch = move_batch(batch, device)
        inputs = batch["x"] if mode == "complex_only" else ablation_features(batch["x"], mode)
        logits = model(inputs)
        loss = criterion_cls(logits, batch["class_id"])
        total_loss += float(loss.detach().cpu())
        n_batches += 1
        targets.extend(batch["class_id"].detach().cpu().tolist())
        logits_list.append(logits.detach().cpu())
    logits_all = torch.cat(logits_list, dim=0)
    metrics = classification_metrics(targets, logits_all)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def train_classifier_ablation(
    name: str,
    mode: str,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    device: torch.device,
) -> Dict[str, float]:
    in_channels = {"magnitude": 4, "phase": 4, "mag_phase": 8, "real_imag": 8}.get(mode)
    if mode == "complex_only":
        ab_model: nn.Module = ComplexOnlyClassifier(num_classes=len(SATELLITE_CLASSES)).to(device)
    else:
        assert in_channels is not None
        ab_model = RealFeatureClassifier(in_channels=in_channels, num_classes=len(SATELLITE_CLASSES)).to(device)

    optimizer_ab = torch.optim.AdamW(ab_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler_ab = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ab, T_max=epochs)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_f1 = -1.0
    best_state = None
    ab_dir = Path(cfg.output_dir) / "ablations"
    ab_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAblation {name}: mode={mode}, epochs={epochs}", flush=True)
    for epoch in range(epochs):
        ab_model.train()
        correct = 0
        seen = 0
        running_loss = 0.0
        total_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            inputs = batch["x"] if mode == "complex_only" else ablation_features(batch["x"], mode)
            optimizer_ab.zero_grad(set_to_none=True)
            logits = ab_model(inputs)
            loss = criterion_cls(logits, batch["class_id"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ab_model.parameters(), cfg.grad_clip_norm)
            optimizer_ab.step()

            running_loss += float(loss.detach().cpu())
            preds = logits.detach().argmax(dim=1)
            correct += int((preds == batch["class_id"]).sum().item())
            seen += int(batch["class_id"].numel())
            if batch_idx == 1 or batch_idx == total_batches or batch_idx % max(cfg.log_every_batches, 1) == 0:
                print(
                    f"{name} epoch {epoch + 1} [{batch_idx:04d}/{total_batches:04d}] "
                    f"loss={running_loss / batch_idx:.4f} acc={correct / max(seen, 1):.3f}",
                    flush=True,
                )

        scheduler_ab.step()
        valid_metrics = evaluate_classifier_ablation(ab_model, valid_loader, mode, device)
        monitored = valid_metrics.get("f1_macro", valid_metrics["accuracy"])
        print(
            f"{name} epoch {epoch + 1} valid_acc={valid_metrics['accuracy']:.4f} "
            f"valid_f1={valid_metrics.get('f1_macro', float('nan')):.4f}",
            flush=True,
        )
        if monitored > best_f1:
            best_f1 = monitored
            best_state = {k: v.detach().cpu().clone() for k, v in ab_model.state_dict().items()}
            torch.save(
                {"model": best_state, "mode": mode, "classes": SATELLITE_CLASSES, "valid": valid_metrics},
                ab_dir / f"{name}_best.pt",
            )

    if best_state is not None:
        ab_model.load_state_dict(best_state)
    test_metrics = evaluate_classifier_ablation(ab_model, test_loader, mode, device)
    result = {"name": name, "mode": mode, "best_valid_f1": best_f1, **{f"test_{k}": v for k, v in test_metrics.items()}}
    print("Ablation result:", json.dumps(result, indent=2), flush=True)
    return result


ablation_epochs = 10
ablation_plan = [
    ("magnitude_only", "magnitude"),
    ("phase_only", "phase"),
    ("magnitude_phase", "mag_phase"),
    ("real_imag", "real_imag"),
    ("complex_only", "complex_only"),
]

ablation_results = []
for ablation_name, ablation_mode in ablation_plan:
    result = train_classifier_ablation(
        ablation_name,
        ablation_mode,
        train_loader,
        valid_loader,
        test_loader,
        epochs=ablation_epochs,
        device=device,
    )
    ablation_results.append(result)

ablation_dir = Path(cfg.output_dir) / "ablations"
ablation_dir.mkdir(parents=True, exist_ok=True)
(ablation_dir / "ablation_results.json").write_text(json.dumps(ablation_results, indent=2))
print("Saved ablation results to:", ablation_dir / "ablation_results.json")

try:
    import pandas as pd

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(ablation_dir / "ablation_results.csv", index=False)
    display(ablation_df)
except Exception:
    print(json.dumps(ablation_results, indent=2))

plt.figure(figsize=(9, 5))
names = [row["name"] for row in ablation_results]
scores = [row["test_f1_macro"] for row in ablation_results]
plt.bar(names, scores)
plt.ylim(0, 1.05)
plt.ylabel("Test Macro F1")
plt.title("Ablation Study: Feature Representation Comparison")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(ablation_dir / "ablation_test_f1.png", dpi=200)
plt.show()


# %% [markdown]
# ## README: Feature Selection Study for Guide Review
#
# This section is separate from the CNN ablation study above. The guide asked for
# feature selection by:
#
# 1. statistical methods
# 2. attention-based methods
#
# Methods implemented:
#
# | Group | Method | What it selects | Classifier used for comparison |
# |---|---|---|---|
# | Statistical | ANOVA F-score | handcrafted magnitude, phase, real/imag, and polarimetric ratio features | Logistic Regression |
# | Statistical | Mutual Information | same handcrafted feature pool | Logistic Regression |
# | Statistical / embedded | ExtraTrees importance | same handcrafted feature pool | Logistic Regression |
# | Attention based | classifier-weight top-k on learned fused features | 128 complex-CNN features + 128 magnitude-phase features | Logistic Regression |
# | Attention based | complex SE channel attention summary | learned latent channels emphasized by the final model | analysis only |
#
# Outputs are saved to:
#
# `/content/drive/MyDrive/drdo/outputs/polisar_complex_multitask/feature_selection/`
#
# What to show the guide:
#
# - `feature_selection_comparison.csv`
# - `feature_selection_comparison_f1.png`
# - `statistical_top_features.csv`
# - `attention_selected_features.csv`
# - `attention_channel_weights.csv`
#
# Interpretation rule:
#
# - Statistical methods tell us which interpretable handcrafted radar features
#   are discriminative.
# - Attention methods tell us which learned fused features and latent channels
#   the final Complex CNN relies on.

# %%
def _safe_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


@torch.no_grad()
def extract_statistical_feature_batch(x: torch.Tensor, eps: float = 1e-8) -> Tuple[np.ndarray, List[str]]:
    """Handcrafted Pol-ISAR feature pool for statistical feature selection."""
    x_cpu = x.detach().cpu()
    mag = torch.abs(x_cpu)
    phase = torch.angle(x_cpu) / math.pi
    real = x_cpu.real
    imag = x_cpu.imag

    features: List[np.ndarray] = []
    names: List[str] = []
    for pol_idx, pol in enumerate(POLS):
        mag_p = mag[:, pol_idx].reshape(x_cpu.shape[0], -1).numpy()
        phase_p = phase[:, pol_idx].reshape(x_cpu.shape[0], -1).numpy()
        real_p = real[:, pol_idx].reshape(x_cpu.shape[0], -1).numpy()
        imag_p = imag[:, pol_idx].reshape(x_cpu.shape[0], -1).numpy()

        per_pol = [
            (f"{pol}_mag_mean", mag_p.mean(axis=1)),
            (f"{pol}_mag_std", mag_p.std(axis=1)),
            (f"{pol}_mag_max", mag_p.max(axis=1)),
            (f"{pol}_mag_p90", np.percentile(mag_p, 90, axis=1)),
            (f"{pol}_mag_energy", np.mean(mag_p ** 2, axis=1)),
            (f"{pol}_phase_mean", phase_p.mean(axis=1)),
            (f"{pol}_phase_std", phase_p.std(axis=1)),
            (f"{pol}_phase_sin_mean", np.sin(np.pi * phase_p).mean(axis=1)),
            (f"{pol}_phase_cos_mean", np.cos(np.pi * phase_p).mean(axis=1)),
            (f"{pol}_real_mean", real_p.mean(axis=1)),
            (f"{pol}_real_std", real_p.std(axis=1)),
            (f"{pol}_imag_mean", imag_p.mean(axis=1)),
            (f"{pol}_imag_std", imag_p.std(axis=1)),
        ]
        for name, values in per_pol:
            names.append(name)
            features.append(values.astype(np.float32))

    energy = torch.mean(mag ** 2, dim=(-2, -1)).numpy()
    pol_index = {pol: idx for idx, pol in enumerate(POLS)}
    hh = energy[:, pol_index["HH"]]
    hv = energy[:, pol_index["HV"]]
    vh = energy[:, pol_index["VH"]]
    vv = energy[:, pol_index["VV"]]
    co = hh + vv
    cross = hv + vh
    ratio_features = [
        ("co_pol_energy", co),
        ("cross_pol_energy", cross),
        ("cross_to_co_energy_ratio", cross / (co + eps)),
        ("hh_to_vv_energy_ratio", hh / (vv + eps)),
        ("hv_to_vh_energy_ratio", hv / (vh + eps)),
        ("hh_minus_vv_energy", hh - vv),
        ("hv_minus_vh_energy", hv - vh),
    ]
    for name, values in ratio_features:
        names.append(name)
        features.append(values.astype(np.float32))

    return np.stack(features, axis=1), names


def extract_statistical_features(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None
    for batch in loader:
        X_batch, names = extract_statistical_feature_batch(batch["x"])
        if feature_names is None:
            feature_names = names
        X_parts.append(X_batch)
        y_parts.append(batch["class_id"].detach().cpu().numpy())
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0), feature_names or []


def standardize_from_train(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (X_train - mean) / std, (X_valid - mean) / std, (X_test - mean) / std


def sklearn_metrics_dict(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }
    try:
        out["auc_ovr"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_ovr"] = float("nan")
    return out


def fit_eval_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", n_jobs=-1)
    clf.fit(X_train, y_train)
    valid_prob = clf.predict_proba(X_valid)
    test_prob = clf.predict_proba(X_test)
    valid_pred = valid_prob.argmax(axis=1)
    test_pred = test_prob.argmax(axis=1)
    return sklearn_metrics_dict(y_valid, valid_prob, valid_pred), sklearn_metrics_dict(y_test, test_prob, test_pred)


def run_statistical_feature_selection(k_values: Sequence[int] = (16, 32, 48, 64)) -> Tuple[List[Dict[str, float]], List[Dict[str, object]]]:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

    X_train, y_train, feature_names = extract_statistical_features(train_loader)
    X_valid, y_valid, _ = extract_statistical_features(valid_loader)
    X_test, y_test, _ = extract_statistical_features(test_loader)
    X_train, X_valid, X_test = standardize_from_train(X_train, X_valid, X_test)

    feature_selection_dir = Path(cfg.output_dir) / "feature_selection"
    feature_selection_dir.mkdir(parents=True, exist_ok=True)
    max_features = X_train.shape[1]
    k_values = [min(int(k), max_features) for k in k_values]
    k_values = sorted(set(k_values))
    results: List[Dict[str, float]] = []
    top_features: List[Dict[str, object]] = []

    def score_selector(selector_name: str, scores: np.ndarray, k: int) -> np.ndarray:
        order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
        selected = order[:k]
        for rank, idx in enumerate(selected, start=1):
            top_features.append(
                {
                    "method": selector_name,
                    "k": k,
                    "rank": rank,
                    "feature": feature_names[int(idx)],
                    "score": float(scores[int(idx)]),
                }
            )
        return selected

    f_selector = SelectKBest(score_func=f_classif, k=max_features).fit(X_train, y_train)
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=max_features).fit(X_train, y_train)
    tree = ExtraTreesClassifier(n_estimators=300, random_state=cfg.seed, class_weight="balanced", n_jobs=-1)
    tree.fit(X_train, y_train)

    selector_scores = {
        "anova_f_score": f_selector.scores_,
        "mutual_information": mi_selector.scores_,
        "extratrees_importance": tree.feature_importances_,
    }

    for method_name, scores in selector_scores.items():
        best_row: Optional[Dict[str, float]] = None
        for k in k_values:
            selected = score_selector(method_name, scores, k)
            valid_metrics, test_metrics = fit_eval_logistic(
                X_train[:, selected],
                y_train,
                X_valid[:, selected],
                y_valid,
                X_test[:, selected],
                y_test,
            )
            row = {
                "group": "statistical",
                "method": method_name,
                "k": float(k),
                "valid_f1_macro": valid_metrics["f1_macro"],
                "valid_accuracy": valid_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_accuracy": test_metrics["accuracy"],
                "test_auc_ovr": test_metrics.get("auc_ovr", float("nan")),
            }
            results.append(row)
            if best_row is None or row["valid_f1_macro"] > best_row["valid_f1_macro"]:
                best_row = row
        print("Best statistical selector:", best_row, flush=True)

    (feature_selection_dir / "statistical_feature_selection_results.json").write_text(json.dumps(results, indent=2))
    (feature_selection_dir / "statistical_top_features.json").write_text(json.dumps(top_features, indent=2))
    try:
        import pandas as pd

        pd.DataFrame(results).to_csv(feature_selection_dir / "statistical_feature_selection_results.csv", index=False)
        pd.DataFrame(top_features).to_csv(feature_selection_dir / "statistical_top_features.csv", index=False)
        display(pd.DataFrame(results))
    except Exception:
        print(json.dumps(results, indent=2), flush=True)
    return results, top_features


statistical_results, statistical_top_features = run_statistical_feature_selection()


# %%
@torch.no_grad()
def collect_fused_vectors(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    vectors: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["x"])
        vectors.append(_safe_numpy(out["fused_vector"]))
        labels.append(_safe_numpy(batch["class_id"]))
    return np.concatenate(vectors, axis=0), np.concatenate(labels, axis=0)


@torch.no_grad()
def summarize_attention_channels(model: ComplexPolISARNet, loader: DataLoader, device: torch.device) -> List[Dict[str, float]]:
    model.eval()
    channel_weights: List[np.ndarray] = []
    spatial_maps: List[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        _ = model(batch["x"])
        if hasattr(model.fusion.channel_attn, "last_weights"):
            channel_weights.append(model.fusion.channel_attn.last_weights.squeeze(-1).squeeze(-1).numpy())
        if hasattr(model.fusion.spatial_attn, "last_map"):
            spatial_maps.append(model.fusion.spatial_attn.last_map.numpy())

    feature_selection_dir = Path(cfg.output_dir) / "feature_selection"
    feature_selection_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []
    if channel_weights:
        weights = np.concatenate(channel_weights, axis=0)
        mean_weights = weights.mean(axis=0)
        for idx, value in enumerate(mean_weights):
            rows.append({"latent_channel": int(idx), "mean_attention_weight": float(value)})
        rows = sorted(rows, key=lambda row: row["mean_attention_weight"], reverse=True)
        try:
            import pandas as pd

            pd.DataFrame(rows).to_csv(feature_selection_dir / "attention_channel_weights.csv", index=False)
        except Exception:
            (feature_selection_dir / "attention_channel_weights.json").write_text(json.dumps(rows, indent=2))

        top_rows = rows[:20]
        plt.figure(figsize=(10, 5))
        plt.bar([str(row["latent_channel"]) for row in top_rows], [row["mean_attention_weight"] for row in top_rows])
        plt.xlabel("Latent channel")
        plt.ylabel("Mean SE attention weight")
        plt.title("Top Complex Attention Channels")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(feature_selection_dir / "attention_top_channels.png", dpi=200)
        plt.show()

    if spatial_maps:
        avg_map = np.concatenate(spatial_maps, axis=0).mean(axis=(0, 1))
        np.save(feature_selection_dir / "attention_average_spatial_map.npy", avg_map)
        plt.figure(figsize=(5, 5))
        plt.imshow(avg_map, cmap="magma")
        plt.colorbar(label="Mean spatial attention")
        plt.title("Average Spatial Attention Map")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(feature_selection_dir / "attention_average_spatial_map.png", dpi=200)
        plt.show()
    return rows


def run_attention_feature_selection(k_values: Sequence[int] = (16, 32, 64, 128)) -> Tuple[List[Dict[str, float]], List[Dict[str, object]]]:
    feature_selection_dir = Path(cfg.output_dir) / "feature_selection"
    feature_selection_dir.mkdir(parents=True, exist_ok=True)

    best_path = Path(cfg.output_dir) / "best_complex_polisar_multitask.pt"
    if best_path.exists():
        load_checkpoint(str(best_path), model, optimizer=None, scheduler=None)
        print("Loaded best Complex CNN checkpoint for attention feature selection:", best_path, flush=True)
    else:
        print("Best checkpoint not found; using current model weights for attention feature selection.", flush=True)

    X_train, y_train = collect_fused_vectors(model, train_loader, device)
    X_valid, y_valid = collect_fused_vectors(model, valid_loader, device)
    X_test, y_test = collect_fused_vectors(model, test_loader, device)
    X_train, X_valid, X_test = standardize_from_train(X_train, X_valid, X_test)

    first_linear = model.classifier[0]
    if not isinstance(first_linear, nn.Linear):
        raise TypeError("Expected model.classifier[0] to be a Linear layer for feature-importance extraction.")
    importance = _safe_numpy(first_linear.weight.abs().mean(dim=0))
    feature_names = [f"complex_vector_{idx:03d}" for idx in range(128)] + [
        f"mag_phase_vector_{idx:03d}" for idx in range(128)
    ]
    order = np.argsort(importance)[::-1]

    results: List[Dict[str, float]] = []
    selected_rows: List[Dict[str, object]] = []
    for k in sorted(set(min(int(k), X_train.shape[1]) for k in k_values)):
        selected = order[:k]
        for rank, idx in enumerate(selected, start=1):
            selected_rows.append(
                {
                    "method": "attention_classifier_weight_topk",
                    "k": k,
                    "rank": rank,
                    "feature": feature_names[int(idx)],
                    "importance": float(importance[int(idx)]),
                }
            )
        valid_metrics, test_metrics = fit_eval_logistic(
            X_train[:, selected],
            y_train,
            X_valid[:, selected],
            y_valid,
            X_test[:, selected],
            y_test,
        )
        row = {
            "group": "attention",
            "method": "attention_classifier_weight_topk",
            "k": float(k),
            "valid_f1_macro": valid_metrics["f1_macro"],
            "valid_accuracy": valid_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_accuracy": test_metrics["accuracy"],
            "test_auc_ovr": test_metrics.get("auc_ovr", float("nan")),
        }
        results.append(row)
        print("Attention feature-selection result:", row, flush=True)

    attention_rows = summarize_attention_channels(model, test_loader, device)
    print("Top attention latent channels:", attention_rows[:10], flush=True)

    (feature_selection_dir / "attention_feature_selection_results.json").write_text(json.dumps(results, indent=2))
    (feature_selection_dir / "attention_selected_features.json").write_text(json.dumps(selected_rows, indent=2))
    try:
        import pandas as pd

        pd.DataFrame(results).to_csv(feature_selection_dir / "attention_feature_selection_results.csv", index=False)
        pd.DataFrame(selected_rows).to_csv(feature_selection_dir / "attention_selected_features.csv", index=False)
        display(pd.DataFrame(results))
    except Exception:
        print(json.dumps(results, indent=2), flush=True)
    return results, selected_rows


attention_results, attention_selected_features = run_attention_feature_selection()


# %%
def save_feature_selection_comparison(
    statistical_results: Sequence[Dict[str, float]],
    attention_results: Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    feature_selection_dir = Path(cfg.output_dir) / "feature_selection"
    feature_selection_dir.mkdir(parents=True, exist_ok=True)
    all_rows = list(statistical_results) + list(attention_results)
    best_rows: List[Dict[str, float]] = []
    for method_name in sorted({row["method"] for row in all_rows}):
        candidates = [row for row in all_rows if row["method"] == method_name]
        best = max(candidates, key=lambda row: row["valid_f1_macro"])
        best_rows.append(best)

    (feature_selection_dir / "feature_selection_comparison.json").write_text(json.dumps(best_rows, indent=2))
    try:
        import pandas as pd

        comparison_df = pd.DataFrame(best_rows).sort_values("test_f1_macro", ascending=False)
        comparison_df.to_csv(feature_selection_dir / "feature_selection_comparison.csv", index=False)
        display(comparison_df)
    except Exception:
        comparison_df = None
        print(json.dumps(best_rows, indent=2), flush=True)

    labels = [row["method"] for row in best_rows]
    scores = [row["test_f1_macro"] for row in best_rows]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores)
    plt.ylim(0, 1.05)
    plt.ylabel("Test Macro F1")
    plt.title("Feature Selection Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(feature_selection_dir / "feature_selection_comparison_f1.png", dpi=200)
    plt.show()
    print("Saved feature selection comparison to:", feature_selection_dir, flush=True)
    return best_rows


feature_selection_comparison = save_feature_selection_comparison(statistical_results, attention_results)


# %% [markdown]
# ## Single-Sample Attention Visualization
#
# This cell creates the figure requested by the guide: one ISAR sample with the
# learned spatial attention map overlaid on the magnitude image.
#
# Output:
#
# `/content/drive/MyDrive/drdo/outputs/polisar_complex_multitask/attention_visualizations/single_sample_attention_overlay.png`

# %%
@torch.no_grad()
def visualize_single_sample_attention(
    sample_index: int = 0,
    split_name: str = "test",
    alpha: float = 0.45,
) -> Path:
    best_path = Path(cfg.output_dir) / "best_complex_polisar_multitask.pt"
    if best_path.exists():
        load_checkpoint(str(best_path), model, optimizer=None, scheduler=None)
        print("Loaded best checkpoint for attention visualization:", best_path, flush=True)
    else:
        print("Best checkpoint not found; using current model weights.", flush=True)

    dataset_map = {"train": train_ds, "valid": valid_ds, "test": test_ds}
    dataset = dataset_map[split_name]
    sample_item = dataset[sample_index]
    x = sample_item["x"].unsqueeze(0).to(device)

    model.eval()
    outputs = model(x)
    probs = torch.softmax(outputs["class_logits"], dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    true_idx = int(sample_item["class_id"])
    pred_name = SATELLITE_CLASSES[pred_idx]
    true_name = SATELLITE_CLASSES[true_idx]

    if not hasattr(model.fusion.spatial_attn, "last_map"):
        raise RuntimeError("Spatial attention map was not captured. Run a forward pass with the updated model.")

    attention_map = model.fusion.spatial_attn.last_map[0, 0].numpy()
    magnitude_image = torch.abs(sample_item["x"]).mean(dim=0).numpy()
    magnitude_image = (magnitude_image - magnitude_image.min()) / (magnitude_image.max() - magnitude_image.min() + 1e-8)

    out_dir = Path(cfg.output_dir) / "attention_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "single_sample_attention_overlay.png"

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(magnitude_image, cmap="gray")
    plt.title(f"ISAR magnitude\nTrue: {true_name}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_image, cmap="gray")
    plt.imshow(attention_map, cmap="jet", alpha=alpha, extent=(0, magnitude_image.shape[1], magnitude_image.shape[0], 0))
    plt.title(f"Attention overlay\nPred: {pred_name} ({probs[pred_idx]:.3f})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.show()

    summary = {
        "split": split_name,
        "sample_index": sample_index,
        "sample_base": sample_item["base"],
        "true_class": true_name,
        "predicted_class": pred_name,
        "predicted_probability": float(probs[pred_idx]),
        "all_probabilities": {SATELLITE_CLASSES[i]: float(probs[i]) for i in range(len(SATELLITE_CLASSES))},
        "figure": str(out_path),
    }
    (out_dir / "single_sample_attention_overlay.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return out_path


attention_overlay_path = visualize_single_sample_attention(sample_index=0, split_name="test")


# %% [markdown]
# ## Cross-Attention Transformer Experiment
#
# This is the next model requested by the guide. It keeps the already trained
# complex CNN backbone, then learns a lightweight transformer classifier over
# two feature streams:
#
# - complex CNN latent vector
# - magnitude-phase vector
#
# The cross-attention layer lets the magnitude-phase stream query the complex
# stream and vice versa. This should be treated as a new experiment, not as a
# replacement for the completed Complex CNN result until it is run and evaluated.
#
# Outputs:
#
# `/content/drive/MyDrive/drdo/outputs/polisar_complex_multitask/cross_attention_transformer/`

# %%
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 128, num_heads: int = 4, dropout: float = 0.10) -> None:
        super().__init__()
        self.complex_queries_mag_phase = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mag_phase_queries_complex = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_complex = nn.LayerNorm(dim)
        self.norm_mag_phase = nn.LayerNorm(dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )

    def forward(self, complex_vector: torch.Tensor, mag_phase_vector: torch.Tensor) -> torch.Tensor:
        complex_token = complex_vector.unsqueeze(1)
        mag_phase_token = mag_phase_vector.unsqueeze(1)
        complex_attended, _ = self.complex_queries_mag_phase(
            query=complex_token,
            key=mag_phase_token,
            value=mag_phase_token,
            need_weights=False,
        )
        mag_phase_attended, _ = self.mag_phase_queries_complex(
            query=mag_phase_token,
            key=complex_token,
            value=complex_token,
            need_weights=False,
        )
        complex_token = self.norm_complex(complex_token + complex_attended)
        mag_phase_token = self.norm_mag_phase(mag_phase_token + mag_phase_attended)
        tokens = torch.cat([complex_token, mag_phase_token], dim=1)
        tokens = self.transformer(tokens)
        return tokens.mean(dim=1)


class CrossAttentionTransformerClassifier(nn.Module):
    def __init__(self, backbone: ComplexPolISARNet, num_classes: int, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.fusion = CrossAttentionFusion(dim=128, num_heads=4, dropout=0.10)
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 96),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(any(param.requires_grad for param in self.backbone.parameters())):
            features = self.backbone(x)
        return self.classifier(self.fusion(features["complex_vector"], features["mag_phase_vector"]))


@torch.no_grad()
def evaluate_cross_attention_classifier(
    model_xattn: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model_xattn.eval()
    logits_parts: List[torch.Tensor] = []
    targets: List[int] = []
    total_loss = 0.0
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.05)
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model_xattn(batch["x"])
        loss = criterion_cls(logits, batch["class_id"])
        total_loss += float(loss.detach().cpu())
        logits_parts.append(logits.detach().cpu())
        targets.extend(batch["class_id"].detach().cpu().tolist())
    metrics = classification_metrics(targets, torch.cat(logits_parts, dim=0))
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def train_cross_attention_transformer(
    epochs: int = 12,
    lr: float = 2e-4,
    freeze_backbone: bool = True,
) -> Dict[str, float]:
    out_dir = Path(cfg.output_dir) / "cross_attention_transformer"
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = ComplexPolISARNet(len(SATELLITE_CLASSES), len(COMPONENT_CLASSES)).to(device)
    best_path = Path(cfg.output_dir) / "best_complex_polisar_multitask.pt"
    if best_path.exists():
        load_checkpoint(str(best_path), backbone, optimizer=None, scheduler=None)
        print("Loaded Complex CNN backbone:", best_path, flush=True)
    else:
        print("Best backbone checkpoint not found; training transformer on current random backbone is not recommended.", flush=True)

    model_xattn = CrossAttentionTransformerClassifier(
        backbone=backbone,
        num_classes=len(SATELLITE_CLASSES),
        freeze_backbone=freeze_backbone,
    ).to(device)
    optimizer_xattn = torch.optim.AdamW(
        [p for p in model_xattn.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler_xattn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_xattn, T_max=epochs)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_f1 = -1.0
    best_state = None
    history_xattn: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model_xattn.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimizer_xattn.zero_grad(set_to_none=True)
            logits = model_xattn(batch["x"])
            loss = criterion_cls(logits, batch["class_id"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_xattn.parameters(), cfg.grad_clip_norm)
            optimizer_xattn.step()
            running_loss += float(loss.detach().cpu())
            preds = logits.detach().argmax(dim=1)
            correct += int((preds == batch["class_id"]).sum().item())
            seen += int(batch["class_id"].numel())
            if batch_idx == 1 or batch_idx == len(train_loader) or batch_idx % max(cfg.log_every_batches, 1) == 0:
                print(
                    f"xattn epoch {epoch + 1} [{batch_idx:04d}/{len(train_loader):04d}] "
                    f"loss={running_loss / batch_idx:.4f} acc={correct / max(seen, 1):.3f}",
                    flush=True,
                )

        scheduler_xattn.step()
        valid_metrics = evaluate_cross_attention_classifier(model_xattn, valid_loader, device)
        row = {
            "epoch": float(epoch),
            "lr": float(scheduler_xattn.get_last_lr()[0]),
            "train_loss": running_loss / max(len(train_loader), 1),
            "train_accuracy": correct / max(seen, 1),
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history_xattn.append(row)
        print(json.dumps(row, indent=2), flush=True)
        if valid_metrics.get("f1_macro", valid_metrics["accuracy"]) > best_f1:
            best_f1 = valid_metrics.get("f1_macro", valid_metrics["accuracy"])
            best_state = {k: v.detach().cpu().clone() for k, v in model_xattn.state_dict().items()}
            torch.save({"model": best_state, "valid": valid_metrics, "epoch": epoch}, out_dir / "best_cross_attention_transformer.pt")

    if best_state is not None:
        model_xattn.load_state_dict(best_state)
    test_metrics = evaluate_cross_attention_classifier(model_xattn, test_loader, device)
    result = {
        "model": "cross_attention_transformer",
        "epochs": float(epochs),
        "freeze_backbone": float(freeze_backbone),
        "best_valid_f1": float(best_f1),
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    (out_dir / "history_cross_attention_transformer.json").write_text(json.dumps(history_xattn, indent=2))
    (out_dir / "results_cross_attention_transformer.json").write_text(json.dumps(result, indent=2))
    print("Cross-attention transformer result:", json.dumps(result, indent=2), flush=True)
    return result


# Run this cell when you want the actual transformer comparison.
# It is intentionally explicit so it does not start a new 12-epoch run by accident.
RUN_CROSS_ATTENTION_TRANSFORMER = False
if RUN_CROSS_ATTENTION_TRANSFORMER:
    cross_attention_transformer_result = train_cross_attention_transformer(epochs=12, lr=2e-4, freeze_backbone=True)
else:
    print("Set RUN_CROSS_ATTENTION_TRANSFORMER=True to train and evaluate the cross-attention transformer.")


# %% [markdown]
# ## Guide-Requested Magnitude/Phase Cross-Attention Classifier Study
#
# This section implements the exact pipeline requested in the latest review:
#
# 1. Extract magnitude feature vector.
# 2. Extract phase feature vector.
# 3. Normalize both streams to `[-1, 1]`.
# 4. Apply cross-attention between magnitude and phase streams.
# 5. Feed attention-based features to:
#    - Transformer classifier
#    - SVM
#    - Random Forest
#    - Softmax / multinomial logistic regression
# 6. Compute classification metrics.
#
# This is intentionally classification-only. The detection head is not part of
# this guide-review pipeline.
#
# Outputs:
#
# `/content/drive/MyDrive/drdo/outputs/polisar_complex_multitask/guide_cross_attention/`

# %%
class MagPhaseVectorizer(nn.Module):
    """Separate magnitude and phase encoders with outputs bounded to [-1, 1]."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.mag_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2, bias=False),
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
        self.phase_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2, bias=False),
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

    def forward(self, x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        mag = torch.abs(x)
        mag_max = mag.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
        mag_norm = 2.0 * (mag / mag_max) - 1.0
        phase_norm = torch.angle(x) / math.pi
        mag_vector = self.mag_encoder(mag_norm.float())
        phase_vector = self.phase_encoder(phase_norm.float())
        return mag_vector, phase_vector


class MagPhaseCrossAttentionTransformer(nn.Module):
    """Cross-attention fusion model for magnitude and phase streams."""

    def __init__(self, num_classes: int, dim: int = 128, num_heads: int = 4, dropout: float = 0.10) -> None:
        super().__init__()
        self.vectorizer = MagPhaseVectorizer(out_dim=dim)
        self.mag_queries_phase = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.phase_queries_mag = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mag = nn.LayerNorm(dim)
        self.norm_phase = nn.LayerNorm(dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 96),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(96, num_classes),
        )
        self.last_attention_weights: Dict[str, torch.Tensor] = {}

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        mag_vector, phase_vector = self.vectorizer(x)
        mag_token = mag_vector.unsqueeze(1)
        phase_token = phase_vector.unsqueeze(1)
        mag_attended, mag_weights = self.mag_queries_phase(
            query=mag_token,
            key=phase_token,
            value=phase_token,
            need_weights=True,
        )
        phase_attended, phase_weights = self.phase_queries_mag(
            query=phase_token,
            key=mag_token,
            value=mag_token,
            need_weights=True,
        )
        self.last_attention_weights = {
            "magnitude_queries_phase": mag_weights.detach().cpu(),
            "phase_queries_magnitude": phase_weights.detach().cpu(),
        }
        mag_token = self.norm_mag(mag_token + mag_attended)
        phase_token = self.norm_phase(phase_token + phase_attended)
        tokens = torch.cat([mag_token, phase_token], dim=1)
        encoded = self.encoder(tokens)
        return encoded.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode_features(x))


@torch.no_grad()
def evaluate_guide_transformer(model_xattn: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model_xattn.eval()
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.05)
    logits_parts: List[torch.Tensor] = []
    targets: List[int] = []
    total_loss = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model_xattn(batch["x"])
        loss = criterion_cls(logits, batch["class_id"])
        total_loss += float(loss.detach().cpu())
        logits_parts.append(logits.detach().cpu())
        targets.extend(batch["class_id"].detach().cpu().tolist())
    metrics = classification_metrics(targets, torch.cat(logits_parts, dim=0))
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


@torch.no_grad()
def collect_guide_attention_features(
    model_xattn: MagPhaseCrossAttentionTransformer,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model_xattn.eval()
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        feats = model_xattn.encode_features(batch["x"])
        features.append(feats.detach().cpu().numpy())
        labels.append(batch["class_id"].detach().cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def run_sklearn_on_attention_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
) -> List[Dict[str, float]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    classifiers = {
        "svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, class_weight="balanced", random_state=cfg.seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            class_weight="balanced",
            random_state=cfg.seed,
            n_jobs=-1,
        ),
        "softmax_logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
        ),
    }

    rows: List[Dict[str, float]] = []
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        valid_prob = clf.predict_proba(X_valid)
        test_prob = clf.predict_proba(X_test)
        valid_pred = valid_prob.argmax(axis=1)
        test_pred = test_prob.argmax(axis=1)
        valid_metrics = sklearn_metrics_dict(y_valid, valid_prob, valid_pred)
        test_metrics = sklearn_metrics_dict(y_test, test_prob, test_pred)
        row = {
            "feature_source": "magnitude_phase_cross_attention",
            "classifier": name,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_f1_macro": valid_metrics["f1_macro"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_auc_ovr": test_metrics.get("auc_ovr", float("nan")),
        }
        rows.append(row)
        print("Attention-feature classifier:", json.dumps(row, indent=2), flush=True)
    return rows


def train_guide_cross_attention_study(epochs: int = 15, lr: float = 3e-4) -> List[Dict[str, float]]:
    out_dir = Path(cfg.output_dir) / "guide_cross_attention"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_xattn = MagPhaseCrossAttentionTransformer(num_classes=len(SATELLITE_CLASSES)).to(device)
    optimizer_xattn = torch.optim.AdamW(model_xattn.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_xattn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_xattn, T_max=epochs)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_f1 = -1.0
    best_state = None
    history_rows_xattn: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model_xattn.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimizer_xattn.zero_grad(set_to_none=True)
            logits = model_xattn(batch["x"])
            loss = criterion_cls(logits, batch["class_id"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_xattn.parameters(), cfg.grad_clip_norm)
            optimizer_xattn.step()

            running_loss += float(loss.detach().cpu())
            preds = logits.detach().argmax(dim=1)
            correct += int((preds == batch["class_id"]).sum().item())
            seen += int(batch["class_id"].numel())
            if batch_idx == 1 or batch_idx == len(train_loader) or batch_idx % max(cfg.log_every_batches, 1) == 0:
                print(
                    f"guide-xattn epoch {epoch + 1} [{batch_idx:04d}/{len(train_loader):04d}] "
                    f"loss={running_loss / batch_idx:.4f} acc={correct / max(seen, 1):.3f}",
                    flush=True,
                )

        scheduler_xattn.step()
        valid_metrics = evaluate_guide_transformer(model_xattn, valid_loader, device)
        row = {
            "epoch": float(epoch + 1),
            "lr": float(scheduler_xattn.get_last_lr()[0]),
            "train_loss": running_loss / max(len(train_loader), 1),
            "train_accuracy": correct / max(seen, 1),
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history_rows_xattn.append(row)
        print(json.dumps(row, indent=2), flush=True)
        monitored = valid_metrics.get("f1_macro", valid_metrics["accuracy"])
        if monitored > best_f1:
            best_f1 = monitored
            best_state = {k: v.detach().cpu().clone() for k, v in model_xattn.state_dict().items()}
            torch.save({"model": best_state, "valid": valid_metrics, "epoch": epoch}, out_dir / "best_mag_phase_cross_attention_transformer.pt")

    if best_state is not None:
        model_xattn.load_state_dict(best_state)

    transformer_test = evaluate_guide_transformer(model_xattn, test_loader, device)
    transformer_row = {
        "feature_source": "magnitude_phase_cross_attention",
        "classifier": "transformer_classifier",
        "valid_f1_macro": float(best_f1),
        "test_accuracy": transformer_test["accuracy"],
        "test_precision_macro": transformer_test.get("precision_macro", float("nan")),
        "test_recall_macro": transformer_test.get("recall_macro", float("nan")),
        "test_f1_macro": transformer_test.get("f1_macro", float("nan")),
        "test_auc_ovr": transformer_test.get("auc_ovr", float("nan")),
        "test_loss": transformer_test["loss"],
    }

    X_train, y_train = collect_guide_attention_features(model_xattn, train_loader, device)
    X_valid, y_valid = collect_guide_attention_features(model_xattn, valid_loader, device)
    X_test, y_test = collect_guide_attention_features(model_xattn, test_loader, device)
    classical_rows = run_sklearn_on_attention_features(X_train, y_train, X_valid, y_valid, X_test, y_test, out_dir)

    all_rows = [transformer_row] + classical_rows
    (out_dir / "guide_cross_attention_history.json").write_text(json.dumps(history_rows_xattn, indent=2))
    (out_dir / "guide_cross_attention_results.json").write_text(json.dumps(all_rows, indent=2))
    try:
        import pandas as pd

        pd.DataFrame(all_rows).to_csv(out_dir / "guide_cross_attention_results.csv", index=False)
        display(pd.DataFrame(all_rows))
    except Exception:
        print(json.dumps(all_rows, indent=2), flush=True)
    print("Saved guide cross-attention study outputs to:", out_dir, flush=True)
    return all_rows


RUN_GUIDE_CROSS_ATTENTION_STUDY = False
if RUN_GUIDE_CROSS_ATTENTION_STUDY:
    guide_cross_attention_results = train_guide_cross_attention_study(epochs=15, lr=3e-4)
else:
    print("Set RUN_GUIDE_CROSS_ATTENTION_STUDY=True to run Transformer/SVM/RF/Softmax metrics.")


# %% [markdown]
# ## Inference Timing

# %%
@torch.no_grad()
def benchmark_inference(model: nn.Module, image_size: int, repeats: int = 100) -> Dict[str, float]:
    model.eval()
    x = torch.randn(1, 4, image_size, image_size, device=device)
    x = torch.complex(x, torch.randn_like(x))
    for _ in range(10):
        _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return {"ms_per_sample": 1000.0 * elapsed / repeats}


print(benchmark_inference(model, cfg.image_size))
