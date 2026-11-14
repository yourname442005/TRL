"""Assessment result dataclasses: criterion statuses, signals, TRLAssessment."""

from __future__ import annotations

from dataclasses import dataclass, field

from trl_intelligence.evidence.models import EvidenceItem

CRITERION_STATUSES = ("satisfied", "weak", "contradicted", "missing")
ASSESSMENT_STATUSES = ("assessed", "insufficient_evidence")
GAP_STATUSES = ("missing", "weak_evidence", "contradicted")
RECOMMENDATION_ACTIONS = ("document", "demonstrate", "test", "analyze")
POLICY_VERSION = "1.0"


@dataclass(frozen=True)
class CriterionStatus:
    criterion_id: str
    trl: int
    status: str
    weight: float
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in CRITERION_STATUSES:
            raise ValueError(f"unknown criterion status: {self.status}")


@dataclass(frozen=True)
class ModelSignal:
    source: str
    predicted_trl: int
    variant: str = "full"

    def __post_init__(self) -> None:
        if not 1 <= self.predicted_trl <= 9:
            raise ValueError(f"signal predicted_trl must be in 1..9, got {self.predicted_trl}")


@dataclass(frozen=True)
class TRLAssessment:
    document_id: str
    status: str
    predicted_trl: int | None
    next_trl: int | None
    confidence: float
    confidence_band: str
    support_scores: dict[str, float]
    supported_criteria: tuple[str, ...]
    missing_criteria: tuple[str, ...]
    contradictory_evidence: tuple[str, ...]
    criterion_statuses: tuple[CriterionStatus, ...]
    evidence: list[EvidenceItem] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    criteria_set_id: str = ""
    policy_version: str = POLICY_VERSION
    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ASSESSMENT_STATUSES:
            raise ValueError(f"unknown assessment status: {self.status}")


@dataclass(frozen=True)
class Recommendation:
    action: str
    specifics: str
    evidence_types_acceptable: tuple[str, ...]
    example_artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.action not in RECOMMENDATION_ACTIONS:
            raise ValueError(f"unknown recommendation action: {self.action}")


@dataclass(frozen=True)
class GapItem:
    criterion_id: str
    trl: int
    requirement: str
    status: str
    partial_evidence_ids: tuple[str, ...]
    blocker: bool
    recommendation: Recommendation

    def __post_init__(self) -> None:
        if self.status not in GAP_STATUSES:
            raise ValueError(f"unknown gap status: {self.status}")


@dataclass(frozen=True)
class DemonstratedCriterion:
    criterion_id: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class GapAnalysis:
    document_id: str
    current_trl: int | None
    next_trl: int | None
    demonstrated: tuple[DemonstratedCriterion, ...]
    gaps: tuple[GapItem, ...]
    priority_order: tuple[str, ...]
