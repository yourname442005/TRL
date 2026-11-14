"""Pure gap analysis: demonstrated criteria, gaps, and priority ordering."""

from __future__ import annotations

from trl_intelligence.assessment.criteria import CriteriaRegistry, Criterion
from trl_intelligence.assessment.models import (
    CriterionStatus,
    DemonstratedCriterion,
    GapAnalysis,
    GapItem,
    Recommendation,
    TRLAssessment,
)
from trl_intelligence.evidence.models import EvidenceSet

_GAP_STATUS_MAP = {
    "missing": "missing",
    "weak": "weak_evidence",
    "contradicted": "contradicted",
}


def _recommend_action(criterion: Criterion) -> str:
    types = set(criterion.evidence_types)
    if "test" in types:
        return "test"
    if types & {"demonstration", "operation"}:
        return "demonstrate"
    if types & {"analysis", "simulation", "design"}:
        return "analyze"
    return "document"


def _build_gap(
    criterion: Criterion,
    status: CriterionStatus,
    blocker: bool,
) -> GapItem:
    return GapItem(
        criterion_id=criterion.criterion_id,
        trl=criterion.trl,
        requirement=criterion.description,
        status=_GAP_STATUS_MAP[status.status],
        partial_evidence_ids=(status.supporting_evidence_ids + status.contradicting_evidence_ids),
        blocker=blocker,
        recommendation=Recommendation(
            action=_recommend_action(criterion),
            specifics=criterion.guidance,
            evidence_types_acceptable=criterion.evidence_types,
            example_artifacts=criterion.example_artifacts,
        ),
    )


def _gap_candidates(
    assessment: TRLAssessment, registry: CriteriaRegistry
) -> list[tuple[Criterion, bool, CriterionStatus]]:
    statuses = {status.criterion_id: status for status in assessment.criterion_statuses}
    candidates: list[tuple[Criterion, bool, CriterionStatus]] = []
    if assessment.predicted_trl is None:
        for criterion in registry.level(1).criteria:
            status = statuses[criterion.criterion_id]
            if status.status != "satisfied":
                candidates.append((criterion, True, status))
        return candidates
    next_trl = assessment.next_trl
    lower_limit = next_trl if next_trl is not None else 10
    if next_trl is not None:
        for criterion in registry.level(next_trl).criteria:
            status = statuses[criterion.criterion_id]
            if status.status != "satisfied":
                candidates.append((criterion, True, status))
    for level in registry.levels:
        if level.trl >= lower_limit:
            continue
        for criterion in level.criteria:
            status = statuses[criterion.criterion_id]
            if status.status in ("weak", "contradicted"):
                candidates.append((criterion, False, status))
    return candidates


def analyze(
    assessment: TRLAssessment,
    registry: CriteriaRegistry,
    evidence_set: EvidenceSet,
) -> GapAnalysis:
    """Derive demonstrated criteria and prioritized gaps from an assessment."""
    demonstrated = tuple(
        DemonstratedCriterion(
            criterion_id=status.criterion_id,
            evidence_ids=tuple(
                item.evidence_id
                for item in evidence_set.items
                if item.criterion_id == status.criterion_id and item.polarity == "supports"
            ),
        )
        for status in assessment.criterion_statuses
        if status.status == "satisfied"
    )

    gaps = [
        _build_gap(criterion, status, blocker)
        for criterion, blocker, status in _gap_candidates(assessment, registry)
    ]
    weights = {
        criterion.criterion_id: criterion.weight
        for level in registry.levels
        for criterion in level.criteria
    }
    gaps.sort(key=lambda gap: (gap.trl, -weights[gap.criterion_id]))

    return GapAnalysis(
        document_id=assessment.document_id,
        current_trl=assessment.predicted_trl,
        next_trl=assessment.next_trl,
        demonstrated=demonstrated,
        gaps=tuple(gaps),
        priority_order=tuple(gap.criterion_id for gap in gaps),
    )
