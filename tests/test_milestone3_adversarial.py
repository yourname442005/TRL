"""Milestone 3 Stages 5-6: adjacent-level discrimination and leakage/adversarial checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

from trl_intelligence.pipeline import AssessmentRequest, assess_document

CORPUS = Path(__file__).resolve().parents[1] / "data" / "validation"
TRL_NUMBER_RE = re.compile(r"trl\s*\d", re.IGNORECASE)


def _payload(path: Path) -> dict:
    result = assess_document(AssessmentRequest(path=path))
    return json.loads(result.to_json())


def _result(path: Path):
    return assess_document(AssessmentRequest(path=path))


class TestAdjacentLevelDiscrimination:
    def test_pair_documents_differ_by_one_maturity_condition_only(self):
        pair_a = (CORPUS / "pairs" / "pair_a_lab.md").read_text(encoding="utf-8")
        pair_b = (CORPUS / "pairs" / "pair_b_relevant_env.md").read_text(encoding="utf-8")
        assert pair_a.count("\n") == pair_b.count("\n") - 2  # one sentence -> two lines
        # the planned sentence is the only substantive difference
        a_lines = [line for line in pair_a.splitlines() if line.strip()]
        b_lines = [line for line in pair_b.splitlines() if line.strip()]
        only_a = [line for line in a_lines if line not in b_lines]
        only_b = [line for line in b_lines if line not in a_lines]
        assert len(only_a) == 1 and len(only_b) == 2
        assert "planned for next year" in only_a[0]

    def test_lab_vs_relevant_environment_is_distinguished(self):
        a = _payload(CORPUS / "pairs" / "pair_a_lab.md")
        b = _payload(CORPUS / "pairs" / "pair_b_relevant_env.md")
        assert a["status"] == "assessed" and b["status"] == "assessed"
        assert a["predicted_trl"] == 4, a["predicted_trl"]
        assert b["predicted_trl"] == 5, b["predicted_trl"]
        gained = set(b["supported_criteria"]) - set(a["supported_criteria"])
        assert gained == {"T5.C1", "T5.C2"}, gained

    def test_no_explicit_trl_keywords_in_condition_pairs(self):
        for name in ("pair_a_lab.md", "pair_b_relevant_env.md"):
            text = (CORPUS / "pairs" / name).read_text(encoding="utf-8")
            assert not TRL_NUMBER_RE.search(text), name

    def test_adding_trl_number_talk_does_not_change_prediction(self):
        a = _payload(CORPUS / "pairs" / "pair_a_lab.md")
        c = _payload(CORPUS / "pairs" / "pair_c_talk.md")
        assert "TRL 6" in (CORPUS / "pairs" / "pair_c_talk.md").read_text(encoding="utf-8")
        assert c["status"] == a["status"]
        assert c["predicted_trl"] == a["predicted_trl"] == 4
        assert c["confidence"] == a["confidence"]
        assert c["supported_criteria"] == a["supported_criteria"]
        assert c["missing_criteria"] == a["missing_criteria"]
        assert c["next_trl_gaps"] == a["next_trl_gaps"]


class TestLeakageAndAdversarial:
    def test_trl9_claims_without_evidence_forced_to_abstain(self):
        payload = _payload(CORPUS / "adversarial" / "adv_leakage.md")
        text = (CORPUS / "adversarial" / "adv_leakage.md").read_text(encoding="utf-8")
        assert "TRL 9" in text and "TRL 8" in text
        assert payload["status"] == "insufficient_evidence"
        assert payload["predicted_trl"] is None
        assert payload["supported_criteria"] == []

    def test_roadmap_trl_mentions_produce_no_supporting_evidence(self):
        result = _result(CORPUS / "adversarial" / "adv_leakage.md")
        assert result.evidence_set is not None
        assert result.evidence_set.items, "expected at least a prospective roadmap item"
        for item in result.evidence_set.items:
            assert item.polarity != "supports", (item.criterion_id, item.quote)
            assert item.trl_level <= 7

    def test_negated_and_speculative_claims_do_not_raise_maturity(self):
        payload = _payload(CORPUS / "adversarial" / "adv_negated.md")
        assert payload["status"] == "assessed"
        assert payload["predicted_trl"] == 2
        assert payload["contradictory_evidence"], "negated claim must surface as contradiction"

    def test_repeated_and_copied_claims_keep_unique_ids_and_stable_level(self):
        result = _result(CORPUS / "adversarial" / "adv_negated.md")
        assert result.evidence_set is not None and result.assessment is not None
        ids = [item.evidence_id for item in result.evidence_set.items]
        assert len(ids) == len(set(ids)), "evidence ids must stay unique under repetition"
        t1_supporting = [
            item
            for item in result.evidence_set.items
            if item.criterion_id == "T1.C1" and item.polarity == "supports"
        ]
        assert len(t1_supporting) == 3, "repeated claim preserved as distinct traceable spans"
        assert result.assessment.predicted_trl == 2

    def test_other_technology_mentions_are_not_captured_as_evidence(self):
        result = _result(CORPUS / "adversarial" / "adv_negated.md")
        assert result.evidence_set is not None
        for item in result.evidence_set.items:
            assert "lidar" not in item.quote.lower(), item.quote
