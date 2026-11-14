"""TRL evaluation metrics: MAE, ±1 accuracy, macro-F1, QWK, per-class, confusion."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)


def mae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def within_1_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if len(true) == 0:
        return 0.0
    return float(np.mean(np.abs(true - pred) <= 1))


def confusion_matrix_9x9(y_true: Sequence[int], y_pred: Sequence[int]) -> list[list[int]]:
    labels = list(range(1, 10))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm.astype(int).tolist()


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Full metric bundle for TRL 1-9 predictions."""
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(true) == 0:
        raise ValueError("empty predictions")

    labels = list(range(1, 10))
    per_class_f1 = f1_score(true, pred, labels=labels, average=None, zero_division=0)
    per_class = {str(label): float(score) for label, score in zip(labels, per_class_f1)}

    accuracy = float(np.mean(true == pred))
    macro_f1 = float(f1_score(true, pred, labels=labels, average="macro", zero_division=0))
    # QWK undefined for single-class y_true in some edge cases; guard
    if len(np.unique(true)) < 2:
        qwk = float("nan")
    else:
        qwk = float(cohen_kappa_score(true, pred, labels=labels, weights="quadratic"))

    return {
        "mae": mae(true, pred),
        "within_1": within_1_accuracy(true, pred),
        "macro_f1": macro_f1,
        "qwk": qwk,
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion": confusion_matrix_9x9(true, pred),
        "n": len(true),
    }


def relative_mae_delta(mae_full: float, mae_masked: float) -> dict[str, float]:
    """Absolute and relative MAE difference from full-text to masked-text.

    rel_delta = (mae_full - mae_masked) / mae_full
    Positive => masked MAE is lower (improvement).
    """
    if mae_full == 0:
        raise ValueError("mae_full must be non-zero to compute relative delta")
    abs_delta = float(mae_full - mae_masked)
    rel_delta = abs_delta / float(mae_full)
    return {"abs_delta": abs_delta, "rel_delta": rel_delta}
