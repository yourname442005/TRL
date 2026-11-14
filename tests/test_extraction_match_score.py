"""RED tests for criterion matching, scoring, and evidence records (Task 7)."""

from pathlib import Path

import pytest

from trl_intelligence.assessment.criteria import CriteriaRegistry
from trl_intelligence.document.models import Document, Page, Section, TextBlock
from trl_intelligence.evidence.models import EvidenceSet
from trl_intelligence.extraction.base import load_pipeline_config
from trl_intelligence.extraction.candidates import CandidateSpan
from trl_intelligence.extraction.match import Match, match_candidate
from trl_intelligence.extraction.pipeline import DeterministicExtractor
from trl_intelligence.extraction.records import build_evidence_item
from trl_intelligence.extraction.score import score

ROOT = Path(__file__).resolve().parents[1]
CRITERIA_PATH = ROOT / "configs" / "trl_criteria_v1.json"
PIPELINE_PATH = ROOT / "configs" / "pipeline_v1.json"

SUPPORTS_TEXT = (
    "The system was demonstrated in a relevant environment in 2024 "
    "and the results show full functionality."
)


def registry() -> CriteriaRegistry:
    return CriteriaRegistry.from_file(CRITERIA_PATH)


def config() -> dict:
    return load_pipeline_config(PIPELINE_PATH)


def candidate(text: str, start: int = 0) -> CandidateSpan:
    return CandidateSpan(
        text=text,
        page=1,
        block_id="p1:b00",
        section="Results",
        start=start,
        end=start + len(text),
    )


def match_for(text: str, criterion_id: str = "T5.C1", trl: int = 5) -> Match:
    return Match(
        criterion_id=criterion_id,
        trl=trl,
        hit_hints=("relevant environment",),
        span=candidate(text),
    )


def make_document() -> Document:
    paragraphs = [
        SUPPORTS_TEXT,
        "The breadboard was validated in the laboratory environment.",
        "The report is organized as follows.",
    ]
    blocks = [
        TextBlock(
            block_id=f"p1:b{index:02d}",
            kind="paragraph",
            text=paragraph,
            start_char=0,
            end_char=len(paragraph),
        )
        for index, paragraph in enumerate(paragraphs)
    ]
    page = Page(
        page_number=1,
        text="\n\n".join(block.text for block in blocks),
        blocks=blocks,
    )
    sections = [
        Section(
            section_id="s1",
            heading="Approach",
            level=1,
            path=["Approach"],
            page_start=1,
            page_end=1,
            block_ids=["p1:b00"],
        ),
        Section(
            section_id="s2",
            heading="Results",
            level=1,
            path=["Results"],
            page_start=1,
            page_end=1,
            block_ids=["p1:b01", "p1:b02"],
        ),
    ]
    return Document(
        document_id="doc1",
        source_path="synthetic.txt",
        source_type="txt",
        title=None,
        metadata={},
        pages=[page],
        sections=sections,
        warnings=[],
    )


class TestMatch:
    def test_relevant_environment_sentence_matches_t5_c1(self):
        matches = match_candidate(candidate(SUPPORTS_TEXT), registry())
        t5 = [match for match in matches if match.criterion_id == "T5.C1"]
        assert len(t5) == 1
        assert "relevant environment" in t5[0].hit_hints
        assert t5[0].trl == 5

    def test_sentence_without_hints_matches_nothing(self):
        matches = match_candidate(candidate("The report is organized as follows."), registry())
        assert matches == []

    def test_all_hits_grouped_for_one_criterion(self):
        text = "The component test was performed in a relevant environment."
        matches = [
            match
            for match in match_candidate(candidate(text), registry())
            if match.criterion_id == "T5.C1"
        ]
        assert len(matches) == 1
        assert set(matches[0].hit_hints) == {"relevant environment", "component test"}


class TestScore:
    def test_supports_assertion(self):
        scored = score(match_for(SUPPORTS_TEXT), config())
        assert scored.polarity == "supports"
        assert scored.evidence_type in {"demonstration", "test"}
        assert scored.strength == "strong"
        assert scored.confidence >= 0.5
        assert scored.confidence == 0.7

    def test_supports_extra_hints_raise_confidence(self):
        match = Match(
            criterion_id="T5.C1",
            trl=5,
            hit_hints=("relevant environment", "component test", "simulated environment"),
            span=candidate(SUPPORTS_TEXT),
        )
        assert score(match, config()).confidence == pytest.approx(0.8)

    def test_prospective_sentence_is_insufficient(self):
        scored = score(match_for("TRL 5 will be demonstrated next year."), config())
        assert scored.polarity == "insufficient"
        assert scored.confidence == 0.4

    def test_bare_mention_without_assertion_is_insufficient(self):
        scored = score(match_for("The relevant environment is important for the study."), config())
        assert scored.polarity == "insufficient"
        assert scored.confidence == 0.4

    def test_failed_test_is_contradicts(self):
        scored = score(match_for("The component failed the relevant environment test."), config())
        assert scored.polarity == "contradicts"
        assert scored.confidence == 0.6

    def test_negation_beats_prospective(self):
        scored = score(
            match_for("The system has not yet been demonstrated in a relevant environment."),
            config(),
        )
        assert scored.polarity == "contradicts"
        assert scored.confidence == 0.6


class TestRecords:
    def test_evidence_item_fields_populated(self):
        match = match_for(SUPPORTS_TEXT)
        scored = score(match, config())
        item = build_evidence_item(match, scored, make_document())
        assert item.evidence_id
        assert item.quote == SUPPORTS_TEXT
        assert item.normalized == " ".join(SUPPORTS_TEXT.split())
        assert item.extraction_method == "deterministic_v1"
        assert item.criterion_id == "T5.C1"
        assert item.trl_level == 5
        assert item.polarity == "supports"

    def test_evidence_set_carries_criteria_set_id(self):
        document = make_document()
        evidence_set = DeterministicExtractor().extract(document, registry(), config())
        assert evidence_set.criteria_set_id == "hackathon_trl_v1"
        assert evidence_set.document_id == document.document_id


class TestEndToEnd:
    def test_extract_returns_evidence_set_on_two_section_document(self):
        document = make_document()
        evidence_set = DeterministicExtractor().extract(document, registry(), config())
        assert isinstance(evidence_set, EvidenceSet)
        assert len(evidence_set.items) == 2
        t5_items = [item for item in evidence_set.items if item.criterion_id == "T5.C1"]
        assert len(t5_items) == 1
        assert t5_items[0].polarity == "supports"
        assert t5_items[0].strength == "strong"
        assert all(item.extraction_method == "deterministic_v1" for item in evidence_set.items)

    def test_extract_is_deterministic(self):
        document = make_document()
        first = DeterministicExtractor().extract(document, registry(), config())
        second = DeterministicExtractor().extract(document, registry(), config())
        assert first.to_dict() == second.to_dict()
