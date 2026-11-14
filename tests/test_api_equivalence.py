"""Milestone 4 Stage 5: CLI and API must be semantically equivalent adapters.

Given the same document bytes and the same configuration, scripts/assess.py and
POST /api/v1/assess must produce the same assessment decisions. The API may only
add wrapper fields (document metadata, processing info) - never different
scoring or decision logic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trl_intelligence.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "validation"

EQUIVALENCE_CASES = [
    "03_relevant_environment.md",
    "04_operational_demo.md",
    "06_contradictory.md",
    "09_multipage_report.pdf",
    "08_insufficient.md",
    "11_controlled_env_test_report.pdf",
]

SHARED_FIELDS = (
    "status",
    "error",
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
    "signals",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def _cli_payload(path: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "assess.py"), "--doc", str(path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode in (0, 2), proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.parametrize("filename", EQUIVALENCE_CASES)
def test_cli_and_api_produce_equivalent_results(client: TestClient, filename: str):
    path = CORPUS / filename
    data = path.read_bytes()
    content_type = "application/pdf" if path.suffix == ".pdf" else "text/markdown"

    cli = _cli_payload(path)
    response = client.post(
        "/api/v1/assess",
        files={"file": (filename, data, content_type)},
    )
    assert response.status_code == 200
    api = response.json()

    for field in SHARED_FIELDS:
        assert api[field] == cli[field], f"{filename}: field {field} differs"
    assert api["evidence"] == cli["evidence"], f"{filename}: evidence differs"
    # API may only add wrapper keys, not replace assessment keys
    assert set(api) - set(cli) == {"document", "processing"}


def test_malformed_pdf_is_equivalent_in_failure(client: TestClient, tmp_path: Path):
    filename = "broken.pdf"
    data = b"%PDF-1.4 garbage not parseable"
    path = tmp_path / filename
    path.write_bytes(data)
    cli = _cli_payload(path)

    response = client.post(
        "/api/v1/assess",
        files={"file": (filename, data, "application/pdf")},
    )
    assert response.status_code == 200
    api = response.json()
    assert api["status"] == cli["status"] == "failed"
    assert api["predicted_trl"] == cli["predicted_trl"] is None
    assert api["evidence"] == cli["evidence"] == []
