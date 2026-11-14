"""RED tests for candidate segmentation, technical detection, and pipeline config (Task 6)."""

import json
from pathlib import Path

from trl_intelligence.document.models import Document, Page, Section, TextBlock
from trl_intelligence.extraction.base import EvidenceExtractor, load_pipeline_config
from trl_intelligence.extraction.candidates import CandidateSpan, segment, split_sentences
from trl_intelligence.extraction.detect import detect
from trl_intelligence.extraction.pipeline import DeterministicExtractor

PIPELINE_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline_v1.json"

THREE_SENTENCES = (
    "The sensor was calibrated. The breadboard passed all checks. Results were recorded."
)
ABBREVIATION_CASE = (
    "The data was reviewed by e.g. Dr. Smith for completeness. Final checks were done."
)


def make_config() -> dict:
    return load_pipeline_config(PIPELINE_PATH)


def make_document(paragraphs: list[str], section_heading: str = "Results") -> Document:
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
    section = Section(
        section_id="s1",
        heading=section_heading,
        level=1,
        path=[section_heading],
        page_start=1,
        page_end=1,
        block_ids=[block.block_id for block in blocks],
    )
    return Document(
        document_id="doc1",
        source_path="synthetic.txt",
        source_type="txt",
        title=None,
        metadata={},
        pages=[page],
        sections=[section],
        warnings=[],
    )


def make_candidate(text: str) -> CandidateSpan:
    return CandidateSpan(
        text=text, page=1, block_id="p1:b00", section="Results", start=0, end=len(text)
    )


class TestPipelineConfig:
    def test_config_file_has_required_sections(self):
        data = json.loads(PIPELINE_PATH.read_text(encoding="utf-8"))
        assert data["schema_version"]
        for key in (
            "min_candidate_chars",
            "max_candidates_per_section",
            "technical_cues",
            "prospective_cues",
            "negation_cues",
            "assertion_cues",
        ):
            assert key in data["extraction"], key
        for key in ("level_threshold", "criterion_min_confidence", "min_evidence_to_assess"):
            assert key in data["aggregation"], key
        for key in ("enabled", "max_chars", "variant"):
            assert key in data["signal"], key
        assert data["strength_map"]["mention"] == "weak"

    def test_load_pipeline_config_returns_dict_with_extraction(self):
        config = make_config()
        assert isinstance(config, dict)
        assert config["extraction"]["technical_cues"]


class TestSentenceSplit:
    def test_three_sentences_become_three_candidates_with_block_offsets(self):
        document = make_document([THREE_SENTENCES])
        candidates = segment(document, make_config())
        assert len(candidates) == 3
        text = THREE_SENTENCES
        expected = [
            "The sensor was calibrated.",
            "The breadboard passed all checks.",
            "Results were recorded.",
        ]
        for candidate, sentence in zip(candidates, expected):
            assert candidate.text == sentence
            assert candidate.start == text.index(sentence)
            assert candidate.end == text.index(sentence) + len(sentence)
            assert candidate.block_id == "p1:b00"
            assert candidate.page == 1
            assert candidate.section == "Results"

    def test_abbreviation_case_not_split(self):
        document = make_document([ABBREVIATION_CASE])
        candidates = segment(document, make_config())
        assert len(candidates) == 2
        assert candidates[0].text.startswith("The data was reviewed by e.g. Dr. Smith")
        assert candidates[1].text == "Final checks were done."

    def test_split_sentences_directly(self):
        sentences = split_sentences(THREE_SENTENCES)
        assert [text for _, _, text in sentences] == [
            "The sensor was calibrated.",
            "The breadboard passed all checks.",
            "Results were recorded.",
        ]


class TestCandidateFilters:
    def test_short_candidates_dropped(self):
        config = make_config()
        config["extraction"]["min_candidate_chars"] = 20
        document = make_document(["OK. " + THREE_SENTENCES])
        candidates = segment(document, config)
        texts = [candidate.text for candidate in candidates]
        assert "OK." not in texts
        assert len(candidates) == 3

    def test_per_section_cap_with_stable_order(self):
        config = make_config()
        config["extraction"]["max_candidates_per_section"] = 2
        paragraphs = [
            (
                "Alpha sentence one is long enough here. Beta sentence two is also long enough. "
                "Gamma sentence three stays. Delta sentence four stays too."
            )
        ]
        document = make_document(paragraphs)
        candidates = segment(document, config)
        assert len(candidates) == 2
        assert candidates[0].text.startswith("Alpha")
        assert candidates[1].text.startswith("Beta")


class TestDetect:
    def test_keeps_technical_candidate(self):
        kept = detect(
            [make_candidate("The prototype was demonstrated in a relevant environment.")],
            make_config(),
        )
        assert len(kept) == 1

    def test_drops_non_technical_candidate(self):
        kept = detect([make_candidate("The report is organized as follows.")], make_config())
        assert kept == []

    def test_detect_preserves_input_order(self):
        config = make_config()
        candidates = [
            make_candidate("Flight test data was collected for the model."),
            make_candidate("The report is organized as follows."),
            make_candidate("The component was qualified after testing."),
        ]
        kept = detect(candidates, config)
        assert [candidate.text for candidate in kept] == [
            "Flight test data was collected for the model.",
            "The component was qualified after testing.",
        ]


class TestEvidenceExtractorProtocol:
    def test_deterministic_extractor_satisfies_runtime_checkable_protocol(self):
        extractor = DeterministicExtractor()
        assert isinstance(extractor, EvidenceExtractor)
