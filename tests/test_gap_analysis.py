"""RED tests for gap analysis (Milestone 2 Task 9)."""

import pytest

from trl_intelligence.assessment.aggregate import aggregate
from trl_intelligence.assessment.criteria import CriteriaRegistry
from trl_intelligence.assessment.models import (
    RECOMMENDATION_ACTIONS,
    GapAnalysis,
    GapItem,
)
from trl_intelligence.evidence.models import EvidenceItem, EvidenceSet
from trl_intelligence.gap_analysis.analyze import analyze

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
    start: int = 0,
) -> EvidenceItem:
    return EvidenceItem(
        document_id="doc1",
        page_number=1,
        section_heading="Results",
        block_id="p1:b00",
        start_char=start,
        end_char=start + 20,
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


def assessment_for(items: list[EvidenceItem], registry: CriteriaRegistry):
    return aggregate(registry, evidence_set(items), CONFIG)


def satisfied_through(registry: CriteriaRegistry, top_trl: int) -> list[EvidenceItem]:
    items = []
    start = 0
    for trl in range(1, top_trl + 1):
        for criterion in registry.level(trl).criteria:
            items.append(item(criterion.criterion_id, trl, start=start))
            start += 100
    return items


class TestNextTrlGaps:
    def test_unsatisfied_next_level_criteria_are_blocker_gaps(self):
        registry = make_registry()
        assessment = assessment_for(satisfied_through(registry, 5), registry)
        assert assessment.predicted_trl == 5
        assert assessment.next_trl == 6
        gap_analysis = analyze(assessment, registry, evidence_set(assessment.evidence))
        assert isinstance(gap_analysis, GapAnalysis)
        assert gap_analysis.current_trl == 5
        assert gap_analysis.next_trl == 6
        assert [gap.criterion_id for gap in gap_analysis.gaps] == ["T6.C1"]
        gap = gap_analysis.gaps[0]
        assert gap.trl == 6
        assert gap.status == "missing"
        assert gap.blocker is True
        criterion = registry.level(6).criteria[0]
        assert gap.requirement == criterion.description
        assert gap.recommendation.specifics == criterion.guidance
        assert gap.recommendation.evidence_types_acceptable
        assert gap.recommendation.action in RECOMMENDATION_ACTIONS
        assert gap.recommendation.example_artifacts

    def test_every_gap_has_criterion_specific_recommendation(self):
        registry = make_registry(criteria_per_level=3)
        items = satisfied_through(registry, 2)
        start = len(items) * 100
        items.extend(
            [
                item("T3.C1", 3, start=start),
                item("T3.C2", 3, start=start + 100),
                item("T3.C3", 3, polarity="contradicts", confidence=0.9, start=start + 200),
                item("T4.C1", 4, start=start + 300),
                item("T4.C2", 4, start=start + 400),
                item("T4.C3", 4, strength="weak", confidence=0.9, start=start + 500),
            ]
        )
        for criterion in registry.level(5).criteria:
            items.append(item(criterion.criterion_id, 5, start=start + 600))
        assessment = assessment_for(items, registry)
        assert assessment.predicted_trl == 5
        gap_analysis = analyze(assessment, registry, evidence_set(assessment.evidence))
        for gap in gap_analysis.gaps:
            criterion = registry.criterion(gap.criterion_id)
            assert gap.recommendation.specifics == criterion.guidance
            assert gap.recommendation.specifics.strip()


class TestWeakEvidenceGap:
    def test_weak_gap_carries_partial_evidence_ids(self):
        registry = make_registry()
        items = satisfied_through(registry, 5)
        weak_item = item("T6.C1", 6, strength="weak", confidence=0.9, start=10_000)
        items.append(weak_item)
        assessment = assessment_for(items, registry)
        gap_analysis = analyze(assessment, registry, evidence_set(assessment.evidence))
        gaps = {gap.criterion_id: gap for gap in gap_analysis.gaps}
        assert gaps["T6.C1"].status == "weak_evidence"
        assert gaps["T6.C1"].partial_evidence_ids
        assert weak_item.evidence_id in gaps["T6.C1"].partial_evidence_ids


class TestPriorityOrder:
    def test_lower_level_contradicted_ranks_before_higher_missing(self):
        registry = make_registry(criteria_per_level=3)
        items = satisfied_through(registry, 2)
        start = 10_000
        items.extend(
            [
                item("T3.C1", 3, start=start),
                item("T3.C2", 3, start=start + 100),
                item("T3.C3", 3, polarity="contradicts", confidence=0.9, start=start + 200),
                item("T4.C1", 4, start=start + 300),
                item("T4.C2", 4, start=start + 400),
                item("T4.C3", 4, strength="weak", confidence=0.9, start=start + 500),
                item("T5.C1", 5, start=start + 600),
                item("T5.C2", 5, start=start + 700),
                item("T5.C3", 5, start=start + 800),
            ]
        )
        assessment = assessment_for(items, registry)
        assert assessment.predicted_trl == 5
        gap_analysis = analyze(assessment, registry, evidence_set(assessment.evidence))
        order = list(gap_analysis.priority_order)
        assert order.index("T3.C3") < order.index("T6.C1")
        assert order[0] == "T3.C3"
        assert set(order) == {
            "T3.C3",
            "T4.C3",
            "T6.C1",
            "T6.C2",
            "T6.C3",
        }
        gap_ids = [gap.criterion_id for gap in gap_analysis.gaps]
        assert gap_ids == order


class TestAbstainGaps:
    def test_abstain_yields_level_one_blockers(self):
        registry = make_registry()
        assessment = assessment_for([], registry)
        assert assessment.predicted_trl is None
        gap_analysis = analyze(assessment, registry, evidence_set([]))
        assert gap_analysis.next_trl is None
        assert gap_analysis.current_trl is None
        assert [gap.criterion_id for gap in gap_analysis.gaps] == ["T1.C1"]
        assert all(gap.blocker for gap in gap_analysis.gaps)
        assert gap_analysis.priority_order == ("T1.C1",)


class TestDemonstrated:
    def test_demonstrated_lists_satisfied_criteria_with_evidence_ids(self):
        registry = make_registry()
        items = satisfied_through(registry, 5)
        evidence = evidence_set(items)
        assessment = assessment_for(items, registry)
        gap_analysis = analyze(assessment, registry, evidence)
        demonstrated_ids = [entry.criterion_id for entry in gap_analysis.demonstrated]
        assert demonstrated_ids == [f"T{trl}.C1" for trl in range(1, 6)]
        all_evidence_ids = {evidence_item.evidence_id for evidence_item in evidence.items}
        for entry in gap_analysis.demonstrated:
            assert entry.evidence_ids
            assert set(entry.evidence_ids) <= all_evidence_ids


class TestGapItemValidation:
    def test_invalid_gap_status_rejected(self):
        from trl_intelligence.assessment.models import Recommendation

        recommendation = Recommendation(
            action="test",
            specifics="specifics",
            evidence_types_acceptable=("test",),
            example_artifacts=("artifact",),
        )
        with pytest.raises(ValueError):
            GapItem(
                criterion_id="T1.C1",
                trl=1,
                requirement="req",
                status="nope",
                partial_evidence_ids=(),
                blocker=True,
                recommendation=recommendation,
            )

    def test_invalid_action_rejected(self):
        with pytest.raises(ValueError):
            from trl_intelligence.assessment.models import Recommendation

            Recommendation(
                action="vibe",
                specifics="specifics",
                evidence_types_acceptable=("test",),
                example_artifacts=("artifact",),
            )
