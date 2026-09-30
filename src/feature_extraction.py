"""Feature extraction utilities for polarimetric ISAR experiments.

The functions here intentionally avoid dataset-specific train/test decisions.
They only read complex ISAR samples and extract feature vectors.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


POLS = ("HH", "HV", "VH", "VV")


def read_full_pol_dat(image_dir: Path, base: str, image_shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Read HH/HV/VH/VV `.dat` files using interleaved float32 real/imag layout."""
    h, w = image_shape
    complex_data = np.zeros((h, w, 2, 2), dtype=np.complex64)
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            file_path = image_dir / f"{base}_{POLS[idx]}.dat"
            combined = np.fromfile(file_path, dtype=np.float32)
            expected = h * w * 2
            if combined.size != expected:
                raise ValueError(f"Unexpected file size for {file_path}: {combined.size}; expected {expected}")
            real = combined[0::2].reshape(h, w)
            imag = combined[1::2].reshape(h, w)
            complex_data[:, :, i, j] = real + 1j * imag
    return complex_data


def pol_matrix_to_channels(complex_data: np.ndarray) -> np.ndarray:
    """Convert `[H,W,2,2]` matrix to `[4,H,W]` in HH/HV/VH/VV order."""
    return np.stack(
        [
            complex_data[:, :, 0, 0],
            complex_data[:, :, 0, 1],
            complex_data[:, :, 1, 0],
            complex_data[:, :, 1, 1],
        ],
        axis=0,
    ).astype(np.complex64)


def resize_stack(x: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(x).float().unsqueeze(0)
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    return resized.numpy()


def extract_magnitude_phase_vector(channels: np.ndarray, feature_size: int = 64) -> np.ndarray:
    """Return flattened `[mag_HH..VV, phase_HH..VV]` feature vector."""
    mag = np.abs(channels).astype(np.float32)
    phase = (np.angle(channels) / math.pi).astype(np.float32)
    mag_small = resize_stack(mag, feature_size)
    phase_small = resize_stack(phase, feature_size)
    return np.concatenate([mag_small.reshape(-1), phase_small.reshape(-1)]).astype(np.float32)


def power_to_db(value: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(value, eps))


def rcs_feature_names() -> List[str]:
    names: List[str] = []
    for pol in POLS:
        names.extend(
            [
                f"{pol}_sum_power",
                f"{pol}_mean_power",
                f"{pol}_max_power",
                f"{pol}_std_power",
                f"{pol}_sum_power_db",
                f"{pol}_mean_power_db",
                f"{pol}_max_power_db",
            ]
        )
    names.extend(["all_sum_power", "all_mean_power", "all_max_power", "all_sum_power_db", "all_mean_power_db", "all_max_power_db"])
    return names


def extract_rcs_features(channels: np.ndarray) -> Dict[str, float]:
    """Compute image-domain RCS-like relative scattering-power features.

    Absolute calibrated RCS requires radar calibration metadata. These features
    are relative image-domain power summaries: `|I|^2`.
    """
    power = (np.abs(channels) ** 2).astype(np.float64)
    row: Dict[str, float] = {}
    for idx, pol in enumerate(POLS):
        p = power[idx]
        sum_power = float(p.sum())
        mean_power = float(p.mean())
        max_power = float(p.max())
        std_power = float(p.std())
        row[f"{pol}_sum_power"] = sum_power
        row[f"{pol}_mean_power"] = mean_power
        row[f"{pol}_max_power"] = max_power
        row[f"{pol}_std_power"] = std_power
        row[f"{pol}_sum_power_db"] = float(power_to_db(sum_power))
        row[f"{pol}_mean_power_db"] = float(power_to_db(mean_power))
        row[f"{pol}_max_power_db"] = float(power_to_db(max_power))

    all_power = power.reshape(-1)
    all_sum = float(all_power.sum())
    all_mean = float(all_power.mean())
    all_max = float(all_power.max())
    row["all_sum_power"] = all_sum
    row["all_mean_power"] = all_mean
    row["all_max_power"] = all_max
    row["all_sum_power_db"] = float(power_to_db(all_sum))
    row["all_mean_power_db"] = float(power_to_db(all_mean))
    row["all_max_power_db"] = float(power_to_db(all_max))
    return row


def extract_all_features_from_dat(image_dir: Path, base: str, feature_size: int = 64) -> Tuple[np.ndarray, Dict[str, float]]:
    channels = pol_matrix_to_channels(read_full_pol_dat(image_dir, base))
    return extract_magnitude_phase_vector(channels, feature_size), extract_rcs_features(channels)
