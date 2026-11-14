"""RED tests for the TRL aggregation policy (Milestone 2 Task 8)."""

import math

import pytest

from trl_intelligence.assessment.aggregate import aggregate
from trl_intelligence.assessment.criteria import CriteriaRegistry
from trl_intelligence.assessment.models import ModelSignal, TRLAssessment
from trl_intelligence.evidence.models import EvidenceItem, EvidenceSet

CONFIG = {
    "aggregation": {
        "level_threshold": 0.6,
        "criterion_min_confidence": 0.5,
        "min_evidence_to_assess": 3,
    }
}


def make_registry(criteria_per_level: int = 1) -> CriteriaRegistry:
    levels = []
    for trl in range(1, 10):
        criteria = [
            {
                "criterion_id": f"T{trl}.C{index}",
                "description": f"Criterion {index} of level {trl}",
                "match_hints": [f"hint {trl} {index}"],
                "guidance": f"Guidance for T{trl}.C{index}",
                "evidence_types": ["test"],
                "example_artifacts": ["artifact"],
                "min_strength": "moderate",
                "weight": 1.0,
            }
            for index in range(1, criteria_per_level + 1)
        ]
        levels.append({"trl": trl, "name": f"Level {trl}", "criteria": criteria})
    data = {
        "schema_version": "1.0",
        "criteria_set_id": "test_set_v1",
        "source_note": "NASA DoD TechPort test set; not a certification standard.",
        "strength_map": {
            "mention": "weak",
            "design": "moderate",
            "report": "moderate",
            "analysis": "moderate",
            "simulation": "moderate",
            "test": "strong",
            "demonstration": "strong",
            "operation": "strong",
        },
        "levels": levels,
    }
    return CriteriaRegistry.from_dict(data)


def item(
    criterion_id: str,
    trl: int,
    *,
    polarity: str = "supports",
    strength: str = "strong",
    confidence: float = 0.8,
) -> EvidenceItem:
    return EvidenceItem(
        document_id="doc1",
        page_number=1,
        section_heading="Results",
        block_id="p1:b00",
        start_char=trl * 10,
        end_char=trl * 10 + 20,
        quote="quoted evidence text",
        normalized="quoted evidence text",
        criterion_id=criterion_id,
        trl_level=trl,
        evidence_type="test",
        polarity=polarity,
        strength=strength,
        confidence=confidence,
        extraction_method="deterministic_v1",
        extractor_version="1.0.0",
    )


def evidence_set(items: list[EvidenceItem]) -> EvidenceSet:
    return EvidenceSet(
        document_id="doc1",
        items=items,
        extraction_config={},
        criteria_set_id="test_set_v1",
        warnings=[],
    )


def statuses_by_id(assessment: TRLAssessment) -> dict:
    return {status.criterion_id: status for status in assessment.criterion_statuses}


class TestCriterionGate:
    def test_gate_statuses(self):
        registry = make_registry(criteria_per_level=4)
        items = [
            item("T5.C1", 5, polarity="supports", strength="strong", confidence=0.8),
            item("T5.C2", 5, polarity="supports", strength="weak", confidence=0.9),
            item("T5.C3", 5, polarity="contradicts", strength="strong", confidence=0.9),
        ]
        assessment = aggregate(registry, evidence_set(items), CONFIG)
        statuses = statuses_by_id(assessment)
        assert statuses["T5.C1"].status == "satisfied"
        assert statuses["T5.C2"].status == "weak"
        assert statuses["T5.C3"].status == "contradicted"
        assert statuses["T5.C4"].status == "missing"


class TestLevelSupport:
    def test_two_of_three_satisfied(self):
        registry = make_registry(criteria_per_level=3)
        items = [
            item("T1.C1", 1),
            item("T1.C2", 1),
        ]
        assessment = aggregate(registry, evidence_set(items), CONFIG)
        assert assessment.support_scores["1"] == pytest.approx(2 / 3)


class TestMonotoneWalk:
    def test_levels_one_to_five_pass(self):
        registry = make_registry()
        items = [item(f"T{trl}.C1", trl) for trl in range(1, 6)]
        assessment = aggregate(registry, evidence_set(items), CONFIG)
        assert assessment.status == "assessed"
        assert assessment.predicted_trl == 5
        assert assessment.next_trl == 6
        assert set(assessment.support_scores) == {str(trl) for trl in range(1, 10)}
        assert len(assessment.support_scores) == 9


class TestAbstention:
    def test_zero_evidence_abstains(self):
        registry = make_registry()
        assessment = aggregate(registry, evidence_set([]), CONFIG)
        assert assessment.status == "insufficient_evidence"
        assert assessment.predicted_trl is None
        assert assessment.next_trl is None
        assert assessment.confidence is not None


class TestConfidenceFormula:
    def test_pinned_value(self):
        registry = make_registry()
        items = [item(f"T{trl}.C1", trl) for trl in range(1, 4)]
        assessment = aggregate(registry, evidence_set(items), CONFIG)
        volume = min(1.0, math.log1p(3) / math.log1p(10))
        expected = 0.60 * 1.0 + 0.25 * volume + 0.15 * 0.5
        assert assessment.confidence == pytest.approx(expected)
        assert assessment.confidence_band == "high"

    def test_signal_agreement_monotonicity(self):
        registry = make_registry()
        items = [item(f"T{trl}.C1", trl) for trl in range(1, 4)]
        evidence = evidence_set(items)
        agreeing = aggregate(
            registry, evidence, CONFIG, signals=[ModelSignal(source="m1", predicted_trl=3)]
        )
        disagreeing = aggregate(
            registry, evidence, CONFIG, signals=[ModelSignal(source="m1", predicted_trl=9)]
        )
        assert agreeing.confidence > disagreeing.confidence
        assert agreeing.signals[0]["agreement"] == 1.0
        assert disagreeing.signals[0]["agreement"] == 0.0


class TestExplainability:
    def test_supported_criteria_have_backing_evidence(self):
        registry = make_registry()
        items = [item(f"T{trl}.C1", trl) for trl in range(1, 6)]
        assessment = aggregate(registry, evidence_set(items), CONFIG)
        assert assessment.supported_criteria == tuple(f"T{trl}.C1" for trl in range(1, 6))
        for criterion_id in assessment.supported_criteria:
            assert any(
                evidence.criterion_id == criterion_id and evidence.polarity == "supports"
                for evidence in assessment.evidence
            )

    def test_contradiction_surfaces_as_evidence_ids(self):
        registry = make_registry()
        contradicting = item("T1.C1", 1, polarity="contradicts", confidence=0.9)
        assessment = aggregate(registry, evidence_set([contradicting]), CONFIG)
        statuses = statuses_by_id(assessment)
        assert statuses["T1.C1"].status == "contradicted"
        assert contradicting.evidence_id in assessment.contradictory_evidence
