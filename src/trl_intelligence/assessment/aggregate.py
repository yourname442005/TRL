"""TRL aggregation: policy v1 (evidence-derived, monotone walk, abstention)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from trl_intelligence.assessment.criteria import CriteriaRegistry, Criterion, TRLLevel
from trl_intelligence.assessment.models import (
    POLICY_VERSION,
    CriterionStatus,
    ModelSignal,
    TRLAssessment,
)
from trl_intelligence.evidence.models import EvidenceItem, EvidenceSet

_STRENGTH_ORDER = {"weak": 0, "moderate": 1, "strong": 2}
_LEVEL_SEQUENCE = tuple(range(1, 10))
_VOLUME_BASE = math.log1p(10)


def _passes_gate(item: EvidenceItem, criterion: Criterion, criterion_min_confidence: float) -> bool:
    return (
        _STRENGTH_ORDER[item.strength] >= _STRENGTH_ORDER[criterion.min_strength]
        and item.confidence >= criterion_min_confidence
    )


def criterion_status(
    criterion: Criterion,
    items: Sequence[EvidenceItem],
    criterion_min_confidence: float,
) -> CriterionStatus:
    supporting = [
        item
        for item in items
        if item.criterion_id == criterion.criterion_id and item.polarity == "supports"
    ]
    contradicting = [
        item
        for item in items
        if item.criterion_id == criterion.criterion_id and item.polarity == "contradicts"
    ]
    if any(_passes_gate(item, criterion, criterion_min_confidence) for item in supporting):
        status = "satisfied"
    elif contradicting:
        status = "contradicted"
    elif supporting:
        status = "weak"
    else:
        status = "missing"
    return CriterionStatus(
        criterion_id=criterion.criterion_id,
        trl=criterion.trl,
        status=status,
        weight=criterion.weight,
        supporting_evidence_ids=tuple(item.evidence_id for item in supporting),
        contradicting_evidence_ids=tuple(item.evidence_id for item in contradicting),
    )


def level_support(level: TRLLevel, statuses: Sequence[CriterionStatus]) -> float:
    by_id = {status.criterion_id: status for status in statuses}
    total_weight = sum(criterion.weight for criterion in level.criteria)
    if total_weight <= 0:
        return 0.0
    satisfied_weight = sum(
        criterion.weight
        for criterion in level.criteria
        if by_id[criterion.criterion_id].status == "satisfied"
    )
    return satisfied_weight / total_weight


def confidence_band(confidence: float) -> str:
    if confidence < 0.4:
        return "low"
    if confidence < 0.7:
        return "medium"
    return "high"


def aggregate(
    registry: CriteriaRegistry,
    evidence_set: EvidenceSet,
    config: Mapping,
    signals: Sequence[ModelSignal] | None = None,
) -> TRLAssessment:
    """Apply aggregation policy v1 to produce a TRLAssessment."""
    signals = list(signals or [])
    aggregation = config["aggregation"]
    level_threshold = float(aggregation["level_threshold"])
    criterion_min_confidence = float(aggregation["criterion_min_confidence"])
    min_evidence = int(aggregation["min_evidence_to_assess"])

    statuses = tuple(
        criterion_status(criterion, evidence_set.items, criterion_min_confidence)
        for level in registry.levels
        for criterion in level.criteria
    )
    support_scores = {str(level.trl): level_support(level, statuses) for level in registry.levels}

    predicted: int | None = None
    for trl in _LEVEL_SEQUENCE:
        if all(support_scores[str(j)] >= level_threshold for j in range(1, trl + 1)):
            predicted = trl
        else:
            break

    n_criteria = max(1, len(statuses))
    n_contradicted = sum(1 for status in statuses if status.status == "contradicted")
    penalty = 1.0 / (1.0 + 0.5 * n_contradicted / n_criteria)

    has_enough_evidence = len(evidence_set.items) >= min_evidence
    if predicted is None or not has_enough_evidence:
        status = "insufficient_evidence"
        predicted_trl = None
        next_trl = None
        confidence = min(1.0, max(0.0, support_scores["1"] * penalty))
    else:
        status = "assessed"
        predicted_trl = predicted
        next_trl = predicted + 1 if predicted < 9 else None
        n_supporting = sum(1 for item in evidence_set.items if item.polarity == "supports")
        volume = min(1.0, math.log1p(n_supporting) / _VOLUME_BASE)
        if signals:
            agreement = sum(
                1.0 if abs(signal.predicted_trl - predicted) <= 1 else 0.0 for signal in signals
            ) / len(signals)
        else:
            agreement = 0.5
        raw = 0.60 * support_scores[str(predicted)] + 0.25 * volume + 0.15 * agreement
        confidence = min(1.0, max(0.0, raw * penalty))

    signal_records = [
        {
            "source": signal.source,
            "predicted_trl": signal.predicted_trl,
            "agreement": (
                1.0
                if predicted_trl is not None and abs(signal.predicted_trl - predicted_trl) <= 1
                else 0.0
            ),
        }
        for signal in signals
    ]

    supported_criteria = tuple(
        status.criterion_id for status in statuses if status.status == "satisfied"
    )
    missing_criteria = tuple(
        status.criterion_id for status in statuses if status.status == "missing"
    )
    contradictory_evidence = tuple(
        item.evidence_id for item in evidence_set.items if item.polarity == "contradicts"
    )

    return TRLAssessment(
        document_id=evidence_set.document_id,
        status=status,
        predicted_trl=predicted_trl,
        next_trl=next_trl,
        confidence=confidence,
        confidence_band=confidence_band(confidence),
        support_scores=support_scores,
        supported_criteria=supported_criteria,
        missing_criteria=missing_criteria,
        contradictory_evidence=contradictory_evidence,
        criterion_statuses=statuses,
        evidence=list(evidence_set.items),
        signals=signal_records,
        criteria_set_id=registry.criteria_set_id,
        policy_version=POLICY_VERSION,
        config=dict(config),
    )
