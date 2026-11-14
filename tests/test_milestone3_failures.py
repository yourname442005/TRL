"""Milestone 3 Stage 8: controlled CLI failure handling - no tracebacks leak."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSESS = ROOT / "scripts" / "assess.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ASSESS), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
        check=False,
    )


def _assert_controlled(proc: subprocess.CompletedProcess[str]) -> dict:
    assert "Traceback" not in proc.stderr, proc.stderr
    assert proc.returncode in (0, 2), (proc.returncode, proc.stderr)
    payload = json.loads(proc.stdout)
    assert payload["status"] in ("assessed", "insufficient_evidence", "failed")
    return payload


def test_missing_file_fails_controlled(tmp_path: Path):
    proc = _run("--doc", str(tmp_path / "does_not_exist.pdf"))
    payload = _assert_controlled(proc)
    assert payload["status"] == "failed"
    assert payload["error"]
    assert proc.returncode == 2


def test_unsupported_extension_fails_controlled(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    proc = _run("--doc", str(csv_path))
    payload = _assert_controlled(proc)
    assert payload["status"] == "failed"
    assert "unsupported" in (payload["error"] or "").lower()
    assert proc.returncode == 2


def test_malformed_pdf_fails_controlled(tmp_path: Path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 definitely not a real pdf body")
    proc = _run("--doc", str(bad))
    payload = _assert_controlled(proc)
    assert payload["status"] == "failed"
    assert payload["error"]
    assert proc.returncode == 2


def test_image_only_pdf_abstains_controlled():
    proc = _run("--doc", str(ROOT / "data" / "validation" / "10_scanned_image_only.pdf"))
    payload = _assert_controlled(proc)
    assert payload["status"] == "insufficient_evidence"
    assert payload["predicted_trl"] is None
    assert proc.returncode == 0


def test_empty_text_file_abstains_controlled(tmp_path: Path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    proc = _run("--doc", str(empty))
    payload = _assert_controlled(proc)
    assert payload["status"] == "insufficient_evidence"
    assert proc.returncode == 0


def test_extremely_long_document_is_controlled(tmp_path: Path):
    paragraph = (
        "The thermal vacuum campaign covered the payload enclosure across the "
        "planned orbital temperature range, and the measured results were recorded "
        "in the test report series for engineering review.\n\n"
    )
    long_doc = tmp_path / "very_long.txt"
    long_doc.write_text(paragraph * 6000, encoding="utf-8")  # ~1.2 MB
    assert long_doc.stat().st_size > 1_000_000
    proc = _run("--doc", str(long_doc))
    payload = _assert_controlled(proc)
    assert payload["status"] in ("assessed", "insufficient_evidence")
    assert proc.returncode == 0


def test_no_arguments_shows_usage_not_traceback():
    proc = _run()
    assert "Traceback" not in proc.stderr, proc.stderr
    assert proc.returncode == 2
    assert "usage" in proc.stderr.lower()
