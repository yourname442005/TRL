"""End-to-end offline assessment pipeline: load → normalize → extract → aggregate → gaps."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trl_intelligence.assessment.aggregate import aggregate
from trl_intelligence.assessment.criteria import CriteriaRegistry
from trl_intelligence.assessment.models import (
    GapAnalysis,
    ModelSignal,
    TRLAssessment,
)
from trl_intelligence.document.loaders import document_from_text, load_document
from trl_intelligence.document.models import Document
from trl_intelligence.document.normalize import normalize
from trl_intelligence.evidence.models import EvidenceSet
from trl_intelligence.extraction.base import load_pipeline_config
from trl_intelligence.extraction.pipeline import DeterministicExtractor
from trl_intelligence.gap_analysis.analyze import analyze
from trl_intelligence.prediction.signals import baseline_signal

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "pipeline_v1.json"
DEFAULT_CRITERIA_PATH = _REPO_ROOT / "configs" / "trl_criteria_v1.json"


@dataclass(frozen=True)
class AssessmentRequest:
    path: Path | None = None
    text: str | None = None
    source_type: str | None = None
    document_id: str | None = None
    config_path: Path | None = None
    criteria_path: Path | None = None
    signals: list[ModelSignal] = field(default_factory=list)
    signal_model_path: Path | None = None
    signal_variant: str = "full"


@dataclass(frozen=True)
class AssessmentResult:
    document: Document | None
    evidence_set: EvidenceSet | None
    assessment: TRLAssessment | None
    gap_analysis: GapAnalysis | None
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.assessment is not None and self.gap_analysis is not None:
            payload: dict[str, Any] = {
                "status": self.status,
                "error": self.error,
                "document_id": self.assessment.document_id,
                "predicted_trl": self.assessment.predicted_trl,
                "confidence": self.assessment.confidence,
                "confidence_band": self.assessment.confidence_band,
                "support_scores": dict(self.assessment.support_scores),
                "supported_criteria": list(self.assessment.supported_criteria),
                "missing_criteria": list(self.assessment.missing_criteria),
                "contradictory_evidence": list(self.assessment.contradictory_evidence),
                "next_trl": self.assessment.next_trl,
                "next_trl_gaps": [asdict(gap) for gap in self.gap_analysis.gaps],
                "evidence": [item.to_dict() for item in self.assessment.evidence],
                "signals": [dict(record) for record in self.assessment.signals],
                "criteria_set_id": self.assessment.criteria_set_id,
                "policy_version": self.assessment.policy_version,
            }
        else:
            payload = {
                "status": self.status,
                "error": self.error,
                "document_id": (self.document.document_id if self.document is not None else None),
                "predicted_trl": None,
                "confidence": None,
                "confidence_band": None,
                "support_scores": {},
                "supported_criteria": [],
                "missing_criteria": [],
                "contradictory_evidence": [],
                "next_trl": None,
                "next_trl_gaps": [],
                "evidence": [],
                "signals": [],
                "criteria_set_id": None,
                "policy_version": None,
            }
        return _round_floats(payload)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item) for item in value]
    return value


def _resolve_document(request: AssessmentRequest) -> Document:
    if request.path is not None:
        return load_document(request.path, document_id=request.document_id)
    if request.text is not None:
        if request.source_type == "pdf":
            raise ValueError("pdf source_type requires a path, not text")
        return document_from_text(request.text, document_id=request.document_id)
    raise ValueError("AssessmentRequest requires either path or text")


def assess_document(request: AssessmentRequest) -> AssessmentResult:
    """Run the full offline assessment; never raises — failures become status=failed."""
    try:
        config = load_pipeline_config(request.config_path or DEFAULT_CONFIG_PATH)
        registry = CriteriaRegistry.from_file(request.criteria_path or DEFAULT_CRITERIA_PATH)
        document = normalize(_resolve_document(request))
        signals = list(request.signals)
        if request.signal_model_path is not None:
            signals.append(
                baseline_signal(
                    request.signal_model_path,
                    document,
                    variant=request.signal_variant,
                )
            )
        evidence_set = DeterministicExtractor().extract(document, registry, config)
        assessment = aggregate(registry, evidence_set, config, signals=signals)
        gap_analysis = analyze(assessment, registry, evidence_set)
        return AssessmentResult(
            document=document,
            evidence_set=evidence_set,
            assessment=assessment,
            gap_analysis=gap_analysis,
            status=assessment.status,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 — controlled failure path by design
        return AssessmentResult(
            document=None,
            evidence_set=None,
            assessment=None,
            gap_analysis=None,
            status="failed",
            error=str(exc) or type(exc).__name__,
        )
