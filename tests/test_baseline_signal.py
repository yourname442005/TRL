"""RED tests for the baseline model signal adapter (Milestone 2 Task 10)."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from trl_intelligence.assessment.models import ModelSignal
from trl_intelligence.document.models import Document, Page, TextBlock
from trl_intelligence.models.baselines import build_b1
from trl_intelligence.pipeline import AssessmentRequest, assess_document
from trl_intelligence.prediction.artifacts import save_model
from trl_intelligence.prediction.signals import baseline_signal

CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "checkpoints"
    / "baseline_v1"
    / "b1_masked.joblib"
)

RICH_TEXT = """BASIC PRINCIPLES

The basic principles were observed and recorded in the scientific basis experiment report.
The underlying phenomenon was characterized experimentally and the data were recorded.

CONCEPT

The application concept was formulated in a trade study and the experiment results were recorded.
The analytical model provided concept verification and was experimentally verified.

PROOF OF CONCEPT

The proof of concept analyzed the critical function experiment and the analytical results were recorded.
A laboratory demonstration performed a bench test at component level and the data were recorded.

LAB VALIDATION

Component validation of the breadboard was verified in lab testing and results were recorded.
The integration risk assessment was validated and the findings were recorded.

RELEVANT ENVIRONMENT

The component test was performed in a relevant environment and the test results show full functionality.
The representative environment fidelity was justified and the testing analysis was recorded.
"""


def make_document(text: str) -> Document:
    block = TextBlock(
        block_id="p1:b00",
        kind="paragraph",
        text=text,
        start_char=0,
        end_char=len(text),
    )
    page = Page(page_number=1, text=text, blocks=[block])
    return Document(
        document_id="doc1",
        source_path="synthetic.txt",
        source_type="txt",
        title=None,
        metadata={},
        pages=[page],
        sections=[],
        warnings=[],
    )


def fit_tiny_b1(tmp_path: Path) -> Path:
    texts = []
    labels = []
    for trl in range(1, 10):
        for _ in range(2):
            texts.append(
                f"technology readiness level demonstration evidence "
                f"marker{trl} additional{trl} assessment report data"
            )
            labels.append(trl)
    model = build_b1(random_state=0, tfidf_params={"min_df": 1})
    model.fit(texts, labels)
    return save_model(model, tmp_path / "tiny_b1.joblib")


def test_baseline_signal_returns_model_signal(tmp_path: Path):
    model_path = fit_tiny_b1(tmp_path)
    document = make_document("technology readiness level 5 demonstration test evidence")
    signal = baseline_signal(model_path, document, variant="full")
    assert isinstance(signal, ModelSignal)
    assert signal.source
    assert 1 <= signal.predicted_trl <= 9
    assert signal.variant == "full"


def test_masked_variant_masks_trl_in_signal_text(tmp_path: Path):
    document = make_document("The system is TRL 5 according to the plan.")
    calls: list[list[str]] = []

    def fake_predict(model_path, texts):
        calls.append(list(texts))
        return np.array([5])

    with patch(
        "trl_intelligence.prediction.signals.predict_texts",
        side_effect=fake_predict,
    ):
        baseline_signal(tmp_path / "any.joblib", document, variant="full")
        baseline_signal(tmp_path / "any.joblib", document, variant="masked")

    assert len(calls) == 2
    assert "TRL 5" in calls[0][0]
    assert "TRL 5" not in calls[1][0]
    assert "[MASKED]" in calls[1][0]


def test_pipeline_records_signal_and_shifts_confidence():
    without = assess_document(AssessmentRequest(text=RICH_TEXT, source_type="txt"))
    with_signal = assess_document(
        AssessmentRequest(
            text=RICH_TEXT,
            source_type="txt",
            signals=[ModelSignal(source="baseline", predicted_trl=5)],
        )
    )
    assert with_signal.assessment is not None
    assert without.assessment is not None
    assert with_signal.assessment.signals
    assert with_signal.assessment.signals[0]["source"] == "baseline"
    assert with_signal.assessment.signals[0]["agreement"] == 1.0
    assert with_signal.assessment.confidence > without.assessment.confidence


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="M1 checkpoint not available")
def test_real_checkpoint_signal():
    document = make_document("technology readiness level 5 demonstration test evidence")
    signal = baseline_signal(CHECKPOINT, document, variant="masked")
    assert isinstance(signal, ModelSignal)
    assert 1 <= signal.predicted_trl <= 9
