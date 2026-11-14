"""Serialize and load fitted models for offline inference."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np


def save_model(model: object, path: str | Path) -> Path:
    """Persist a fitted estimator with joblib."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    return out


def load_model(path: str | Path) -> object:
    """Load a model from disk (offline)."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"model not found: {src}")
    return joblib.load(src)


def predict_texts(model_or_path: object | str | Path, texts: Sequence[str]) -> np.ndarray:
    """Predict TRL labels for raw text strings."""
    model = load_model(model_or_path) if isinstance(model_or_path, str | Path) else model_or_path
    return np.asarray(model.predict(list(texts)))
