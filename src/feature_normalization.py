"""Train-stat-only feature normalization utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np


EPS = 1e-6


def compute_column_stats(X_train: np.ndarray, eps: float = EPS) -> Dict[str, np.ndarray]:
    """Compute column-wise statistics from training data only."""
    stats = {
        "mean": X_train.mean(axis=0).astype(np.float32),
        "var": X_train.var(axis=0).astype(np.float32),
        "std": (X_train.std(axis=0) + eps).astype(np.float32),
        "min": X_train.min(axis=0).astype(np.float32),
        "max": X_train.max(axis=0).astype(np.float32),
        "median": np.median(X_train, axis=0).astype(np.float32),
        "q25": np.percentile(X_train, 25, axis=0).astype(np.float32),
        "q75": np.percentile(X_train, 75, axis=0).astype(np.float32),
    }
    stats["iqr"] = (stats["q75"] - stats["q25"] + eps).astype(np.float32)
    stats["maxabs"] = np.maximum(np.abs(X_train).max(axis=0), eps).astype(np.float32)
    return stats


def normalize_with_train_stats(X: np.ndarray, stats: Dict[str, np.ndarray], method: str, eps: float = EPS) -> np.ndarray:
    """Apply a normalization using statistics already computed from training data."""
    if method == "raw":
        return X.astype(np.float32)
    if method == "standard_z":
        return ((X - stats["mean"]) / stats["std"]).astype(np.float32)
    if method == "minmax_0_1":
        return ((X - stats["min"]) / (stats["max"] - stats["min"] + eps)).astype(np.float32)
    if method == "minmax_-1_1":
        return (2.0 * (X - stats["min"]) / (stats["max"] - stats["min"] + eps) - 1.0).astype(np.float32)
    if method == "robust_iqr":
        return ((X - stats["median"]) / stats["iqr"]).astype(np.float32)
    if method == "maxabs":
        return (X / stats["maxabs"]).astype(np.float32)
    raise ValueError(f"Unknown normalization method: {method}")


def save_stats_npz(path: str, stats: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **stats)


def load_stats_npz(path: str) -> Dict[str, np.ndarray]:
    loaded = np.load(path)
    return {key: loaded[key] for key in loaded.files}
