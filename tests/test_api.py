"""RED tests for the FastAPI application (Milestone 4 Stage 4)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from trl_intelligence.api.app import create_app

CORPUS = Path(__file__).resolve().parents[1] / "data" / "validation"
RICH_DOC = CORPUS / "03_relevant_environment.md"


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def _make_pdf_bytes(text: str = "Component validation was verified in lab testing.") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


class TestHealth:
    def test_health_returns_service_identity(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["service"] == "trl-intelligence"
        assert body["criteria_set_id"] == "hackathon_trl_v1"
        assert body["policy_version"] == "1.0"
        assert body["criteria_content_hash"]


class TestAssessEndpoint:
    def test_text_markdown_upload_assessed(self, client: TestClient):
        response = client.post(
            "/api/v1/assess",
            files={"file": ("report.md", RICH_DOC.read_bytes(), "text/markdown")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "assessed"
        assert body["predicted_trl"] == 5
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
            "document",
            "processing",
        ):
            assert key in body, key
        assert body["document"]["source_name"] == "report.md"
        assert body["processing"]["processing_time_ms"] >= 0

    def test_pdf_upload_assessed_with_metadata(self, client: TestClient):
        response = client.post(
            "/api/v1/assess",
            files={"file": ("note.pdf", _make_pdf_bytes(), "application/pdf")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("assessed", "insufficient_evidence")
        assert body["document"]["source_type"] == "pdf"
        assert body["document"]["page_count"] == 1

    def test_missing_file_field_is_controlled_422(self, client: TestClient):
        response = client.post("/api/v1/assess")
        assert response.status_code == 422
        body = response.json()
        assert body["status"] == "failed"
        assert body["code"] == "invalid_request"
        assert "Traceback" not in response.text

    def test_unsupported_extension_is_415(self, client: TestClient):
        response = client.post(
            "/api/v1/assess",
            files={"file": ("payload.exe", b"MZ...", "application/octet-stream")},
        )
        assert response.status_code == 415
        assert response.json()["code"] == "unsupported_type"

    def test_bad_content_type_is_415(self, client: TestClient):
        response = client.post(
            "/api/v1/assess",
            files={"file": ("page.txt", b"hello there friend", "text/html")},
        )
        assert response.status_code == 415
        assert response.json()["code"] == "unsupported_content_type"

    def test_empty_file_is_400(self, client: TestClient):
        response = client.post(
            "/api/v1/assess",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "empty_file"

    def test_oversized_file_is_413(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("trl_intelligence.api.service.MAX_UPLOAD_BYTES", 64)
        response = client.post(
            "/api/v1/assess",
            files={"file": ("big.txt", b"x" * 200, "text/plain")},
        )
        assert response.status_code == 413
        assert response.json()["code"] == "file_too_large"

    def test_pdf_without_magic_bytes_is_415(self, client: TestClient):
        response = client.post(
            "/api/v1/assess",
            files={"file": ("fake.pdf", b"<html>nope</html>", "application/pdf")},
        )
        assert response.status_code == 415
        assert response.json()["code"] == "invalid_content"

    def test_malformed_pdf_returns_200_with_controlled_failed_payload(
        self, client: TestClient
    ):
        response = client.post(
            "/api/v1/assess",
            files={"file": ("bad.pdf", b"%PDF-1.4 garbage", "application/pdf")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "failed"
        assert body["error"]
        assert "Traceback" not in response.text

    def test_missing_or_non_file_field_is_controlled(self, client: TestClient):
        response = client.post(
            "/api/v1/assess",
            files={"file": (None, b"some content here", "text/plain")},
        )
        assert response.status_code == 422
        body = response.json()
        assert body["status"] == "failed"
        assert body["code"] == "invalid_request"
        assert "Traceback" not in response.text

    def test_internal_error_is_controlled_500_without_traceback(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        def boom(*_args, **_kwargs):
            raise RuntimeError("secret internal detail")

        monkeypatch.setattr(
            "trl_intelligence.api.service.AssessmentService.assess_upload", boom
        )
        response = client.post(
            "/api/v1/assess",
            files={"file": ("report.md", b"content content", "text/markdown")},
        )
        assert response.status_code == 500
        body = response.json()
        assert body["status"] == "failed"
        assert body["code"] == "internal_error"
        assert "Traceback" not in response.text
        assert "secret internal detail" not in response.text


class TestNoArbitraryPathAccess:
    def test_assess_endpoint_rejects_get_with_path_query(self, client: TestClient):
        response = client.get("/api/v1/assess", params={"path": "/etc/passwd"})
        assert response.status_code == 405
        assert "Traceback" not in response.text

    def test_unknown_paths_are_404_json(self, client: TestClient):
        response = client.get("/etc/passwd")
        assert response.status_code == 404
        body = response.json()
        assert body["status"] == "failed"
        assert body["code"] == "not_found"

    def test_upload_ignores_extra_path_fields(self, client: TestClient):
        response = client.post(
            "/api/v1/assess",
            data={"path": "/etc/passwd", "config": "../../secrets.json"},
            files={"file": ("report.md", RICH_DOC.read_bytes(), "text/markdown")},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "assessed"
