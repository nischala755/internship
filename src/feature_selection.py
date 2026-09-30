"""Feature selection helpers for extracted ISAR feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
except Exception:  # pragma: no cover - lets notebooks give a clear runtime error
    SelectKBest = None
    f_classif = mutual_info_classif = None
    RandomForestClassifier = None


@dataclass
class SelectionResult:
    method: str
    indices: np.ndarray
    scores: np.ndarray


def select_anova_f(X_train: np.ndarray, y_train: np.ndarray, k: int = 512) -> SelectionResult:
    if SelectKBest is None or f_classif is None:
        raise RuntimeError("scikit-learn is required for ANOVA feature selection.")
    selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
    selector.fit(X_train, y_train)
    indices = selector.get_support(indices=True)
    scores = np.nan_to_num(selector.scores_[indices], nan=0.0, posinf=0.0, neginf=0.0)
    return SelectionResult("anova_f", indices.astype(np.int64), scores.astype(np.float32))


def select_mutual_info(X_train: np.ndarray, y_train: np.ndarray, k: int = 512, random_state: int = 42) -> SelectionResult:
    if mutual_info_classif is None:
        raise RuntimeError("scikit-learn is required for mutual information feature selection.")
    scores_all = mutual_info_classif(X_train, y_train, random_state=random_state)
    indices = np.argsort(scores_all)[::-1][: min(k, X_train.shape[1])]
    return SelectionResult("mutual_info", indices.astype(np.int64), scores_all[indices].astype(np.float32))


def select_random_forest_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 512,
    random_state: int = 42,
    n_estimators: int = 200,
) -> SelectionResult:
    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn is required for random forest feature selection.")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    scores_all = clf.feature_importances_
    indices = np.argsort(scores_all)[::-1][: min(k, X_train.shape[1])]
    return SelectionResult("random_forest_importance", indices.astype(np.int64), scores_all[indices].astype(np.float32))


def apply_selection(X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return X[:, indices]


def save_selection(path: str, result: SelectionResult) -> None:
    np.savez_compressed(path, method=result.method, indices=result.indices, scores=result.scores)


def load_selection(path: str) -> SelectionResult:
    loaded = np.load(path, allow_pickle=True)
    return SelectionResult(str(loaded["method"]), loaded["indices"].astype(np.int64), loaded["scores"].astype(np.float32))


def summarize_selection(result: SelectionResult) -> Dict[str, float | str | int]:
    return {
        "method": result.method,
        "num_features": int(result.indices.size),
        "top_index": int(result.indices[0]) if result.indices.size else -1,
        "top_score": float(result.scores[0]) if result.scores.size else 0.0,
        "mean_score": float(result.scores.mean()) if result.scores.size else 0.0,
    }
