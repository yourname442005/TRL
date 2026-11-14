"""Real-world regression: equivalent technical phrasing must yield traceable evidence.

The fixture is a controlled-environment validation report whose key statements
are phrased differently from the criteria match hints (for example a sentence
about validation demonstrating an integrated prototype under controlled
laboratory conditions, where no hint literal appears verbatim). Extraction must
recognize such statements, must not inflate roadmap or limitation statements
into support, and must abstain when the research-foundation criteria are
genuinely absent from the text. No expected TRL level is asserted here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trl_intelligence.pipeline import AssessmentRequest, assess_document

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "validation" / "11_controlled_env_test_report.pdf"


@pytest.fixture(scope="module")
def payload() -> dict:
    return assess_document(AssessmentRequest(path=FIXTURE)).to_dict()


class TestEquivalentPhrasingExtraction:
    def test_produces_traceable_evidence(self, payload: dict) -> None:
        assert payload["error"] is None
        assert len(payload["evidence"]) >= 10
        for item in payload["evidence"]:
            assert item["criterion_id"]
            assert item["quote"].strip()
            assert item["confidence"] > 0

    def test_equivalent_phrases_support_the_criteria_they_evidence(
        self, payload: dict
    ) -> None:
        supported = set(payload["supported_criteria"])
        assert {"T3.C2", "T4.C1", "T6.C1", "T6.C2"} <= supported, supported

    def test_research_foundation_criteria_are_genuinely_absent(
        self, payload: dict
    ) -> None:
        missing = set(payload["missing_criteria"])
        for criterion_id in ("T1.C1", "T1.C2", "T2.C1", "T2.C2"):
            assert criterion_id in missing, criterion_id


class TestNoInflation:
    def test_roadmap_and_deployment_statements_never_support(
        self, payload: dict
    ) -> None:
        supported = set(payload["supported_criteria"])
        assert "T9.C1" not in supported
        assert "T5.C1" not in supported

    def test_explicit_not_yet_limitation_surfaces_as_contradiction(
        self, payload: dict
    ) -> None:
        assert payload["contradictory_evidence"]

    def test_absent_research_foundation_forces_abstention(self, payload: dict) -> None:
        assert payload["status"] == "insufficient_evidence"
        assert payload["predicted_trl"] is None
