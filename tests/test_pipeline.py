"""RED tests for the end-to-end assessment pipeline (Milestone 2 Task 10)."""

import json
from pathlib import Path

from trl_intelligence.pipeline import (
    AssessmentRequest,
    AssessmentResult,
    assess_document,
)

RICH_TEXT = """BASIC PRINCIPLES

The basic principles were observed and recorded in the scientific basis experiment report.
The underlying phenomenon was characterized experimentally and the data were recorded.

CONCEPT

The application concept was formulated in a trade study and the experiment results were recorded.
The analytical model provided concept verification and was experimentally verified.

PROOF OF CONCEPT

The proof of concept analyzed the critical function experiment and the analytical results were recorded.
A laboratory demonstration performed a bench test at component level and the data were recorded.

LAB VALIDATION

Component validation of the breadboard was verified in lab testing and results were recorded.
The integration risk assessment was validated and the findings were recorded.

RELEVANT ENVIRONMENT

The component test was performed in a relevant environment and the test results show full functionality.
The representative environment fidelity was justified and the testing analysis was recorded.
"""


def test_assessed_on_rich_fixture():
    result = assess_document(AssessmentRequest(text=RICH_TEXT, source_type="txt"))
    assert isinstance(result, AssessmentResult)
    assert result.status == "assessed"
    assert result.error is None
    assert result.document is not None
    assert result.evidence_set is not None
    assert result.assessment is not None
    assert result.gap_analysis is not None
    assert result.assessment.predicted_trl == 5
    assert result.assessment.next_trl == 6


def test_one_liner_abstains():
    result = assess_document(AssessmentRequest(text="Hello world.", source_type="txt"))
    assert result.status == "insufficient_evidence"
    assert result.assessment is not None
    assert result.assessment.predicted_trl is None


def test_empty_text_abstains():
    result = assess_document(AssessmentRequest(text="", source_type="txt"))
    assert result.status == "insufficient_evidence"


def test_malformed_pdf_fails_gracefully(tmp_path: Path):
    bad_pdf = tmp_path / "malformed.pdf"
    bad_pdf.write_bytes(b"this is definitely not a pdf file")
    result = assess_document(AssessmentRequest(path=bad_pdf))
    assert result.status == "failed"
    assert result.error
    assert result.document is None


def test_to_json_round_trips_and_is_stable():
    result = assess_document(AssessmentRequest(text=RICH_TEXT, source_type="txt"))
    first = result.to_json()
    second = result.to_json()
    assert first == second
    payload = json.loads(first)
    assert isinstance(payload, dict)


def test_result_json_contains_required_keys():
    result = assess_document(AssessmentRequest(text=RICH_TEXT, source_type="txt"))
    payload = json.loads(result.to_json())
    for key in (
        "predicted_trl",
        "confidence",
        "supported_criteria",
        "missing_criteria",
        "contradictory_evidence",
        "next_trl",
        "next_trl_gaps",
        "evidence",
    ):
        assert key in payload, key
    assert payload["predicted_trl"] == 5


def test_pdf_evidence_page_numbers(tmp_path: Path):
    import pymupdf

    pdf_path = tmp_path / "two_pages.pdf"
    document = pymupdf.open()
    page_one = document.new_page()
    page_one.insert_text((72, 72), "RESULTS", fontsize=12)
    page_one.insert_text(
        (72, 110),
        "The component test was performed in a relevant environment "
        "and the test results show full functionality.",
        fontsize=9,
    )
    page_two = document.new_page()
    page_two.insert_text(
        (72, 72),
        "The flight test was completed and the results show full functionality.",
        fontsize=9,
    )
    document.save(str(pdf_path))
    document.close()

    result = assess_document(AssessmentRequest(path=pdf_path))
    assert result.status != "failed", result.error
    assert result.evidence_set is not None
    pages = {item.page_number for item in result.evidence_set.items}
    assert 1 in pages, pages
    assert 2 in pages, pages
