"""Milestone 4 Stage 9: end-to-end test over a real socket.

Spawns a real uvicorn process serving the FastAPI app, talks to it with a real
HTTP client, and checks that the wire protocol matches the CLI adapter.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "validation"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_server():
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "trl_intelligence.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20.0
    while True:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"uvicorn failed to start:\n{output}")
        try:
            response = httpx.get(f"{base}/health", timeout=0.5)
            if response.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            proc.kill()
            raise RuntimeError("uvicorn did not become healthy in 20s")
        time.sleep(0.1)
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def test_health_over_real_socket(live_server: str):
    response = httpx.get(f"{live_server}/health", timeout=5)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "trl-intelligence"
    assert body["criteria_set_id"] == "hackathon_trl_v1"


def test_assess_matches_cli_over_real_socket(live_server: str):
    path = CORPUS / "03_relevant_environment.md"
    data = path.read_bytes()

    cli_proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "assess.py"), "--doc", str(path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert cli_proc.returncode == 0, cli_proc.stderr
    cli = json.loads(cli_proc.stdout)

    response = httpx.post(
        f"{live_server}/api/v1/assess",
        files={"file": ("03_relevant_environment.md", data, "text/markdown")},
        timeout=30,
    )
    assert response.status_code == 200
    api = response.json()

    for field in (
        "status",
        "predicted_trl",
        "confidence",
        "document_id",
        "supported_criteria",
        "missing_criteria",
        "next_trl",
        "criteria_set_id",
        "policy_version",
    ):
        assert api[field] == cli[field], f"field {field} differs over the wire"
    assert api["evidence"] == cli["evidence"]


def test_pdf_upload_over_real_socket(live_server: str):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Component validation was verified in lab testing.")
    data = doc.tobytes()
    doc.close()

    response = httpx.post(
        f"{live_server}/api/v1/assess",
        files={"file": ("lab.pdf", data, "application/pdf")},
        timeout=30,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("assessed", "insufficient_evidence")
    assert body["document"]["source_type"] == "pdf"
    assert body["document"]["page_count"] == 1


def test_controlled_error_over_real_socket(live_server: str):
    response = httpx.post(
        f"{live_server}/api/v1/assess",
        files={"file": ("x.exe", b"MZ", "application/octet-stream")},
        timeout=10,
    )
    assert response.status_code == 415
    body = response.json()
    assert body["status"] == "failed"
    assert body["code"] == "unsupported_type"
    assert "Traceback" not in response.text


def test_malformed_pdf_over_real_socket(live_server: str):
    response = httpx.post(
        f"{live_server}/api/v1/assess",
        files={"file": ("bad.pdf", b"%PDF-1.4 garbage", "application/pdf")},
        timeout=10,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"]
    assert "Traceback" not in response.text
