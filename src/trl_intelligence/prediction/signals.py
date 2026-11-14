"""Baseline classifier signal adapter: optional frozen-model signal (M1 read-only)."""

from __future__ import annotations

from pathlib import Path

from trl_intelligence.assessment.models import ModelSignal
from trl_intelligence.document.models import Document
from trl_intelligence.prediction.artifacts import predict_texts
from trl_intelligence.preprocessing.text import mask_trl

__all__ = ["ModelSignal", "baseline_signal"]


def document_signal_text(document: Document) -> str:
    return "\n".join(page.text for page in document.pages)


def baseline_signal(
    model_path: str | Path,
    document: Document,
    variant: str = "full",
) -> ModelSignal:
    """Predict a TRL signal using the frozen M1 baseline model (read-only)."""
    text = document_signal_text(document)
    if variant == "masked":
        text = mask_trl(text)
    prediction = predict_texts(model_path, [text])
    return ModelSignal(
        source=f"baseline:{Path(model_path).stem}",
        predicted_trl=int(prediction[0]),
        variant=variant,
    )
