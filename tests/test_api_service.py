"""RED tests for the thin assessment service (Milestone 4 Stage 3).

The service wraps the existing pipeline without duplicating any assessment logic
and must be independently testable without HTTP.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pymupdf
import pytest

from trl_intelligence.api.service import (
    MAX_UPLOAD_BYTES,
    AssessmentService,
    ServiceValidationError,
)
from trl_intelligence.pipeline import AssessmentRequest, assess_document

CORPUS = Path(__file__).resolve().parents[1] / "data" / "validation"
RICH_DOC = CORPUS / "03_relevant_environment.md"


def _make_pdf_bytes(text: str = "Component validation was verified in lab testing.") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


class TestAssessUpload:
    def test_text_upload_returns_full_response_payload(self):
        service = AssessmentService()
        payload = service.assess_upload(
            RICH_DOC.read_bytes(), "report.md", "text/markdown"
        )
        assert payload["status"] == "assessed"
        assert payload["predicted_trl"] == 5
        for key in (
            "confidence",
            "confidence_band",
            "supported_criteria",
            "missing_criteria",
            "contradictory_evidence",
            "next_trl",
            "next_trl_gaps",
            "evidence",
            "document_id",
            "criteria_set_id",
            "policy_version",
        ):
            assert key in payload, key

    def test_document_metadata_and_processing_present(self):
        service = AssessmentService()
        payload = service.assess_upload(
            RICH_DOC.read_bytes(), "report.md", "text/markdown"
        )
        document = payload["document"]
        assert document["source_name"] == "report.md"
        assert document["source_type"] == "md"
        assert document["page_count"] == 1
        assert document["document_id"] == payload["document_id"]
        assert document["loader"] == "markdown"
        processing = payload["processing"]
        assert processing["processing_time_ms"] > 0
        assert processing["extractor_version"] == "1.0.0"

    def test_document_id_is_deterministic_for_same_bytes(self):
        service = AssessmentService()
        data = RICH_DOC.read_bytes()
        first = service.assess_upload(data, "a.md", "text/markdown")
        second = service.assess_upload(data, "a.md", "text/markdown")
        assert first["document_id"] == second["document_id"]

    def test_pdf_upload_assessed_with_page_count(self):
        service = AssessmentService()
        payload = service.assess_upload(_make_pdf_bytes(), "note.pdf", "application/pdf")
        assert payload["status"] in ("assessed", "insufficient_evidence")
        assert payload["status"] != "failed", payload.get("error")
        assert payload["document"]["page_count"] == 1
        assert payload["document"]["source_type"] == "pdf"

    def test_malformed_pdf_returns_controlled_failed_payload(self):
        service = AssessmentService()
        payload = service.assess_upload(b"%PDF-1.4 not really a pdf", "bad.pdf", "application/pdf")
        assert payload["status"] == "failed"
        assert payload["error"]
        assert "Traceback" not in str(payload)


class TestEquivalenceWithPipeline:
    def test_service_payload_matches_direct_pipeline_result(self):
        service = AssessmentService()
        data = RICH_DOC.read_bytes()
        payload = service.assess_upload(data, "report.md", "text/markdown")
        direct = assess_document(AssessmentRequest(path=RICH_DOC)).to_dict()
        for key in (
            "status",
            "document_id",
            "predicted_trl",
            "confidence",
            "confidence_band",
            "support_scores",
            "supported_criteria",
            "missing_criteria",
            "contradictory_evidence",
            "next_trl",
            "next_trl_gaps",
            "criteria_set_id",
            "policy_version",
        ):
            assert payload[key] == direct[key], key
        assert [item["evidence_id"] for item in payload["evidence"]] == [
            item["evidence_id"] for item in direct["evidence"]
        ]


class TestInputValidation:
    def test_unsupported_extension_rejected(self, tmp_path: Path):
        service = AssessmentService()
        with pytest.raises(ServiceValidationError) as exc:
            service.assess_upload(b"whatever", "payload.exe", "application/octet-stream")
        assert exc.value.code == "unsupported_type"

    def test_empty_file_rejected(self):
        service = AssessmentService()
        with pytest.raises(ServiceValidationError) as exc:
            service.assess_upload(b"", "empty.txt", "text/plain")
        assert exc.value.code == "empty_file"

    def test_oversized_file_rejected(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("trl_intelligence.api.service.MAX_UPLOAD_BYTES", 64)
        service = AssessmentService()
        with pytest.raises(ServiceValidationError) as exc:
            service.assess_upload(b"x" * 65, "big.txt", "text/plain")
        assert exc.value.code == "file_too_large"
        assert MAX_UPLOAD_BYTES >= 1_000_000  # default stays a real-world limit

    def test_bad_content_type_rejected(self):
        service = AssessmentService()
        with pytest.raises(ServiceValidationError) as exc:
            service.assess_upload(b"<html></html>", "page.txt", "text/html")
        assert exc.value.code == "unsupported_content_type"

    def test_pdf_without_magic_bytes_rejected(self):
        service = AssessmentService()
        with pytest.raises(ServiceValidationError) as exc:
            service.assess_upload(b"definitely not a pdf", "fake.pdf", "application/pdf")
        assert exc.value.code == "invalid_content"

    def test_path_traversal_filename_is_neutralized(self):
        service = AssessmentService()
        payload = service.assess_upload(
            _make_pdf_bytes(), "../../etc/passwd.pdf", "application/pdf"
        )
        assert payload["status"] != "failed" or payload["error"]
        assert payload["document"]["source_name"] == "passwd.pdf"

    def test_no_temp_files_left_behind(self, monkeypatch: pytest.MonkeyPatch):
        created: list[Path] = []
        real_mkdtemp = tempfile.mkdtemp

        def spy(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs)
            created.append(Path(path))
            return path

        monkeypatch.setattr(tempfile, "mkdtemp", spy)
        service = AssessmentService()
        service.assess_upload(RICH_DOC.read_bytes(), "report.md", "text/markdown")
        assert created, "service should stage uploads in a temp directory"
        for path in created:
            assert not path.exists(), f"temp dir not cleaned up: {path}"
