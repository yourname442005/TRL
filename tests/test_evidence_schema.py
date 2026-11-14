"""RED tests for the evidence schema (Milestone 2 Task 5)."""

import hashlib
import json

import pytest

from trl_intelligence.assessment.models import (
    CRITERION_STATUSES,
    CriterionStatus,
)
from trl_intelligence.evidence.models import (
    EVIDENCE_TYPES,
    POLARITIES,
    STRENGTHS,
    EvidenceItem,
    EvidenceSet,
    make_evidence_id,
)

LONG_QUOTE = "x" * 400


def make_item(**overrides) -> EvidenceItem:
    defaults = {
        "document_id": "doc1",
        "page_number": 1,
        "section_heading": "Results",
        "block_id": "p1:b00",
        "start_char": 0,
        "end_char": 54,
        "quote": "The prototype was demonstrated in a relevant environment.",
        "normalized": "The prototype was demonstrated in a relevant environment.",
        "criterion_id": "T5.C1",
        "trl_level": 5,
        "evidence_type": "demonstration",
        "polarity": "supports",
        "strength": "strong",
        "confidence": 0.8,
        "extraction_method": "deterministic_v1",
        "extractor_version": "1.0.0",
    }
    defaults.update(overrides)
    return EvidenceItem(**defaults)


def make_set(items: list[EvidenceItem] | None = None) -> EvidenceSet:
    return EvidenceSet(
        document_id="doc1",
        items=list(items or []),
        extraction_config={"knob": 1},
        criteria_set_id="hackathon_trl_v1",
        warnings=[],
    )


class TestEnums:
    def test_polarities(self):
        assert POLARITIES == ("supports", "contradicts", "insufficient")

    def test_evidence_types(self):
        assert EVIDENCE_TYPES == (
            "mention",
            "design",
            "report",
            "analysis",
            "simulation",
            "test",
            "demonstration",
            "operation",
        )

    def test_strengths(self):
        assert STRENGTHS == ("weak", "moderate", "strong")

    def test_criterion_statuses(self):
        assert CRITERION_STATUSES == ("satisfied", "weak", "contradicted", "missing")


class TestEvidenceItemValidation:
    def test_trl_level_below_one_rejected(self):
        with pytest.raises(ValueError):
            make_item(trl_level=0)

    def test_trl_level_above_nine_rejected(self):
        with pytest.raises(ValueError):
            make_item(trl_level=10)

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValueError):
            make_item(confidence=-0.1)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValueError):
            make_item(confidence=1.1)

    def test_invalid_polarity_rejected(self):
        with pytest.raises(ValueError):
            make_item(polarity="maybe")

    def test_invalid_evidence_type_rejected(self):
        with pytest.raises(ValueError):
            make_item(evidence_type="anecdote")

    def test_invalid_strength_rejected(self):
        with pytest.raises(ValueError):
            make_item(strength="extreme")


class TestEvidenceId:
    def test_identical_location_and_criteria_give_equal_ids(self):
        assert make_item().evidence_id == make_item().evidence_id

    def test_different_criterion_gives_different_id(self):
        other = make_item(criterion_id="T5.C2")
        assert make_item().evidence_id != other.evidence_id

    def test_different_page_gives_different_id(self):
        other = make_item(page_number=2)
        assert make_item().evidence_id != other.evidence_id

    def test_matches_documented_formula(self):
        expected = hashlib.sha1(b"doc1:1:p1:b00:0:T5.C1").hexdigest()[:16]
        assert make_item().evidence_id == expected
        assert (
            make_evidence_id(
                document_id="doc1",
                page_number=1,
                block_id="p1:b00",
                start_char=0,
                criterion_id="T5.C1",
            )
            == expected
        )


class TestQuoteClipping:
    def test_long_quote_clipped_to_300_with_ellipsis(self):
        item = make_item(quote=LONG_QUOTE, end_char=400)
        assert len(item.quote) == 300
        assert item.quote == "x" * 299 + "…"

    def test_short_quote_untouched(self):
        quote = "The prototype was demonstrated in a relevant environment."
        assert make_item(quote=quote).quote == quote


class TestEvidenceSet:
    def test_sort_orders_by_trl_criterion_page_start(self):
        items = [
            make_item(criterion_id="T5.C2", page_number=2, start_char=10),
            make_item(criterion_id="T5.C1", page_number=1, start_char=5),
            make_item(criterion_id="T4.C1", trl_level=4, page_number=3, start_char=0),
            make_item(criterion_id="T5.C1", page_number=1, start_char=0),
        ]
        evidence_set = make_set(items)
        evidence_set.sort()
        keys = [
            (item.trl_level, item.criterion_id, item.page_number, item.start_char)
            for item in evidence_set.items
        ]
        assert keys == [
            (4, "T4.C1", 3, 0),
            (5, "T5.C1", 1, 0),
            (5, "T5.C1", 1, 5),
            (5, "T5.C2", 2, 10),
        ]

    def test_add_keeps_items_sorted(self):
        evidence_set = make_set()
        evidence_set.add(make_item(criterion_id="T5.C2", page_number=9))
        evidence_set.add(make_item(criterion_id="T4.C1", trl_level=4, page_number=1))
        assert [item.criterion_id for item in evidence_set.items] == ["T4.C1", "T5.C2"]

    def test_to_dict_is_json_serializable(self):
        evidence_set = make_set()
        evidence_set.add(make_item())
        payload = evidence_set.to_dict()
        text = json.dumps(payload)
        restored = json.loads(text)
        assert restored["criteria_set_id"] == "hackathon_trl_v1"
        assert restored["document_id"] == "doc1"
        assert restored["items"][0]["evidence_id"] == evidence_set.items[0].evidence_id
        assert restored["extraction_config"] == {"knob": 1}


class TestCriterionStatus:
    def test_valid_status_accepted(self):
        status = CriterionStatus(
            criterion_id="T5.C1",
            trl=5,
            status="satisfied",
            weight=1.0,
            supporting_evidence_ids=("abc123",),
            contradicting_evidence_ids=(),
        )
        assert status.status == "satisfied"

    def test_invalid_status_rejected(self):
        with pytest.raises(ValueError):
            CriterionStatus(
                criterion_id="T5.C1",
                trl=5,
                status="nope",
                weight=1.0,
                supporting_evidence_ids=(),
                contradicting_evidence_ids=(),
            )
