"""Milestone 3 end-to-end validation: golden envelopes, traceability, determinism.

Every expected result in data/validation/envelopes.json is derived from the
documented evidence in the corpus files, not from observed pipeline output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trl_intelligence.assessment.aggregate import _STRENGTH_ORDER
from trl_intelligence.assessment.criteria import CriteriaRegistry
from trl_intelligence.document.loaders import load_document
from trl_intelligence.evidence.models import QUOTE_MAX_CHARS, make_evidence_id
from trl_intelligence.extraction.base import load_pipeline_config
from trl_intelligence.pipeline import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CRITERIA_PATH,
    AssessmentRequest,
    AssessmentResult,
    assess_document,
)

CORPUS = Path(__file__).resolve().parents[1] / "data" / "validation"
ENVELOPES = json.loads((CORPUS / "envelopes.json").read_text(encoding="utf-8"))
CASES = ENVELOPES["cases"]
CASE_IDS = [case["id"] for case in CASES]
BAND_RANK = {"low": 0, "medium": 1, "high": 2}

_CACHE: dict[str, AssessmentResult] = {}


def _case(case_id: str) -> dict:
    return next(case for case in CASES if case["id"] == case_id)


def _result(case_id: str) -> AssessmentResult:
    if case_id not in _CACHE:
        case = _case(case_id)
        _CACHE[case_id] = assess_document(AssessmentRequest(path=CORPUS / case["file"]))
    return _CACHE[case_id]


def _payload(case_id: str) -> dict:
    return json.loads(_result(case_id).to_json())


class TestGoldenEnvelopes:
    @pytest.mark.parametrize("case_id", CASE_IDS)
    def test_status_matches_envelope(self, case_id: str):
        case, payload = _case(case_id), _payload(case_id)
        assert payload["status"] == case["expected_status"], payload.get("error")

    @pytest.mark.parametrize("case_id", CASE_IDS)
    def test_predicted_trl_matches_envelope(self, case_id: str):
        case, payload = _case(case_id), _payload(case_id)
        expected = case["expected_trl"]
        assert payload["predicted_trl"] == expected
        rng = case.get("acceptable_trl_range")
        if expected is not None and rng:
            assert rng[0] <= payload["predicted_trl"] <= rng[1]

    @pytest.mark.parametrize("case_id", CASE_IDS)
    def test_confidence_band_meets_minimum(self, case_id: str):
        case, payload = _case(case_id), _payload(case_id)
        band = payload["confidence_band"]
        assert BAND_RANK[band] >= BAND_RANK[case["min_confidence_band"]], (
            f"{band} < {case['min_confidence_band']} (confidence={payload['confidence']})"
        )

    @pytest.mark.parametrize("case_id", CASE_IDS)
    def test_required_supported_criteria_present(self, case_id: str):
        case, payload = _case(case_id), _payload(case_id)
        supported = set(payload["supported_criteria"])
        missing = [cid for cid in case["required_supported_criteria"] if cid not in supported]
        assert not missing, f"required supported criteria absent: {missing}"

    @pytest.mark.parametrize("case_id", CASE_IDS)
    def test_next_level_blocker_gaps_match_envelope(self, case_id: str):
        case, payload = _case(case_id), _payload(case_id)
        if case["expected_status"] != "assessed":
            return
        blockers = {gap["criterion_id"] for gap in payload["next_trl_gaps"] if gap.get("blocker")}
        assert blockers == set(case["expected_missing_next_criteria"])

    @pytest.mark.parametrize("case_id", CASE_IDS)
    def test_abstention_behavior(self, case_id: str):
        case, payload = _case(case_id), _payload(case_id)
        if case["expect_abstention"]:
            assert payload["predicted_trl"] is None
            assert payload["status"] == "insufficient_evidence"
        else:
            assert payload["predicted_trl"] is not None

    @pytest.mark.parametrize("case_id", CASE_IDS)
    def test_contradiction_flag_matches_envelope(self, case_id: str):
        case, payload = _case(case_id), _payload(case_id)
        if case["expected_contradictions"]:
            assert payload["contradictory_evidence"], "contradictions silently discarded"
        else:
            assert payload["contradictory_evidence"] == []


class TestTraceability:
    @pytest.mark.parametrize("case_id", CASE_IDS)
    def test_every_evidence_id_resolves_to_source_span(self, case_id: str):
        result = _result(case_id)
        assert result.document is not None and result.evidence_set is not None
        blocks = {
            block.block_id: (page.page_number, block)
            for page in result.document.pages
            for block in page.blocks
        }
        for item in result.evidence_set.items:
            assert item.block_id in blocks, item.evidence_id
            page_number, block = blocks[item.block_id]
            assert page_number == item.page_number, item.evidence_id
            source_slice = block.text[item.start_char : item.end_char]
            if len(source_slice) > QUOTE_MAX_CHARS:
                assert item.quote == source_slice[: QUOTE_MAX_CHARS - 1] + "…"
            else:
                assert item.quote == source_slice, item.evidence_id
            assert item.evidence_id == make_evidence_id(
                document_id=item.document_id,
                page_number=item.page_number,
                block_id=item.block_id,
                start_char=item.start_char,
                criterion_id=item.criterion_id,
            )

    @pytest.mark.parametrize("case_id", CASE_IDS)
    def test_quotes_are_present_in_page_text(self, case_id: str):
        result = _result(case_id)
        assert result.document is not None and result.evidence_set is not None
        for item in result.evidence_set.items:
            page = result.document.pages[item.page_number - 1]
            assert item.quote in page.text, (
                f"quote not on page {item.page_number}: {item.quote[:60]!r}"
            )

    @pytest.mark.parametrize("case_id", CASE_IDS)
    def test_supported_criteria_have_backing_evidence_ids(self, case_id: str):
        result = _result(case_id)
        assert result.assessment is not None and result.evidence_set is not None
        evidence_ids = {item.evidence_id for item in result.evidence_set.items}
        statuses = {s.criterion_id: s for s in result.assessment.criterion_statuses}
        for cid in result.assessment.supported_criteria:
            status = statuses[cid]
            assert status.status == "satisfied"
            assert status.supporting_evidence_ids, cid
            for eid in status.supporting_evidence_ids:
                assert eid in evidence_ids, (cid, eid)
        for demonstrated in result.gap_analysis.demonstrated:
            assert demonstrated.evidence_ids, demonstrated.criterion_id

    @pytest.mark.parametrize("case_id", CASE_IDS)
    def test_missing_criteria_are_genuinely_absent_or_insufficient(self, case_id: str):
        result = _result(case_id)
        assert result.assessment is not None
        registry = CriteriaRegistry.from_file(DEFAULT_CRITERIA_PATH)
        config = load_pipeline_config(DEFAULT_CONFIG_PATH)
        min_conf = float(config["aggregation"]["criterion_min_confidence"])
        statuses = {s.criterion_id: s for s in result.assessment.criterion_statuses}
        items = {item.evidence_id: item for item in result.assessment.evidence}

        for cid in result.assessment.missing_criteria:
            status = statuses[cid]
            assert status.status == "missing", cid
            assert not status.supporting_evidence_ids, cid

        case = _case(case_id)
        if case["expected_status"] != "assessed":
            return
        for cid in case["expected_missing_next_criteria"]:
            status = statuses[cid]
            assert status.status != "satisfied", cid
            if status.status == "weak":
                criterion = registry.criterion(cid)
                for eid in status.supporting_evidence_ids:
                    item = items[eid]
                    strength_ok = (
                        _STRENGTH_ORDER[item.strength] >= _STRENGTH_ORDER[criterion.min_strength]
                    )
                    assert not (strength_ok and item.confidence >= min_conf), (
                        f"{cid} marked weak but {eid} passes the gate"
                    )

    @pytest.mark.parametrize("case_id", CASE_IDS)
    def test_contradicting_items_resolve_to_contradicts_polarity(self, case_id: str):
        payload = _payload(case_id)
        if not payload["contradictory_evidence"]:
            return
        result = _result(case_id)
        items = {item.evidence_id: item for item in result.evidence_set.items}
        for eid in payload["contradictory_evidence"]:
            assert eid in items, eid
            assert items[eid].polarity == "contradicts", eid


class TestDeterminism:
    @pytest.mark.parametrize("case_id", CASE_IDS)
    def test_repeated_runs_produce_identical_json(self, case_id: str):
        case = _case(case_id)
        request = AssessmentRequest(path=CORPUS / case["file"])
        first = assess_document(request).to_json()
        second = assess_document(request).to_json()
        assert first == second


class TestMultipagePdfPages:
    def test_expected_evidence_pages_present(self):
        payload = _payload("multipage_report")
        case = _case("multipage_report")
        pages = sorted({item["page_number"] for item in payload["evidence"]})
        assert pages == case["expected_evidence_pages"]

    def test_repeated_header_and_page_numbers_are_stripped(self):
        result = _result("multipage_report")
        document = result.document
        assert document is not None
        for page in document.pages:
            lines = [line for line in page.text.split("\n") if line.strip()]
            assert not any("MERIDIAN AEROSPACE" in line for line in lines), page.page_number
        assert document.metadata["removed_repeated_lines"] >= 5
        assert document.metadata["removed_page_number_lines"] >= 5


class TestScannedPdf:
    def test_scanned_fixture_loads_with_no_text_warning(self):
        document = load_document(CORPUS / "10_scanned_image_only.pdf")
        assert "no_extractable_text" in document.warnings
        assert "possibly_scanned_image_only" in document.warnings

    def test_scanned_fixture_abstains_without_error(self):
        payload = _payload("scanned_image_only")
        assert payload["status"] == "insufficient_evidence"
        assert payload["error"] is None
        assert payload["predicted_trl"] is None
