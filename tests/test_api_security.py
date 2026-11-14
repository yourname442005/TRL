"""Milestone 4 Stage 8: upload security tests."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from trl_intelligence.api.app import create_app

CORPUS = Path(__file__).resolve().parents[1] / "data" / "validation"


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def _pdf_bytes() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Component validation was verified in lab testing.")
    data = doc.tobytes()
    doc.close()
    return data


class TestNoShellExecution:
    def test_upload_path_never_invokes_a_shell(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        def forbid(*_args, **_kwargs):
            raise AssertionError("shell execution attempted during assessment")

        for target in ("os.system", "os.popen", "subprocess.run", "subprocess.Popen", "subprocess.call"):
            monkeypatch.setattr(target, forbid)

        response = client.post(
            "/api/v1/assess",
            files={
                "file": (
                    "report; rm -rf $HOME.md",
                    (CORPUS / "03_relevant_environment.md").read_bytes(),
                    "text/markdown",
                )
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "assessed"


class TestSafeTemporaryFiles:
    def test_staged_filename_is_not_user_controlled(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        captured: list[tuple[Path, list[str]]] = []
        real_rmtree = shutil.rmtree

        def spy(path, *args, **kwargs):
            directory = Path(path)
            names = [entry.name for entry in directory.iterdir()] if directory.exists() else []
            captured.append((directory, names))
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", spy)

        response = client.post(
            "/api/v1/assess",
            files={"file": ("../../evil.pdf", _pdf_bytes(), "application/pdf")},
        )
        assert response.status_code == 200
        assert captured, "expected a staged temp directory"
        directory, names = captured[0]
        assert directory.parent.resolve() == Path(tempfile.gettempdir()).resolve()
        assert names, "staged file should exist before cleanup"
        assert all(name.startswith("upload.") for name in names), names
        assert not directory.exists(), "temp directory must be cleaned up"
        assert not (Path.cwd() / "evil.pdf").exists()
        assert not (Path.cwd() / "passwd.pdf").exists()

    def test_absolute_path_filename_is_neutralized(self, client: TestClient):
        response = client.post(
            "/api/v1/assess",
            files={"file": ("/etc/passwd.pdf", _pdf_bytes(), "application/pdf")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["document"]["source_name"] == "passwd.pdf"


class TestErrorBodiesNeverLeakInternals:
    @pytest.mark.parametrize(
        ("files", "data", "expected_status"),
        [
            ({"file": ("x.exe", b"MZ", "application/octet-stream")}, None, 415),
            ({"file": ("e.txt", b"", "text/plain")}, None, 400),
            ({"file": ("t.txt", b"hello there", "text/html")}, None, 415),
            ({"file": ("f.pdf", b"<html>", "application/pdf")}, None, 415),
            ({}, None, 422),
        ],
    )
    def test_error_responses_are_clean_json(
        self, client: TestClient, files, data, expected_status
    ):
        del data
        response = client.post("/api/v1/assess", files=files)
        assert response.status_code == expected_status
        body = response.json()
        assert body["status"] == "failed"
        assert isinstance(body["error"], str) and body["error"]
        for leak in ("Traceback", "site-packages", "/Users/", "File \""):
            assert leak not in response.text, leak

    def test_oversized_upload_error_is_clean(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("trl_intelligence.api.service.MAX_UPLOAD_BYTES", 32)
        response = client.post(
            "/api/v1/assess",
            files={"file": ("big.txt", b"y" * 100, "text/plain")},
        )
        assert response.status_code == 413
        assert "Traceback" not in response.text

    def test_internal_error_is_clean(self, client: TestClient, monkeypatch: pytest.MonkeyPatch):
        def boom(*_args, **_kwargs):
            raise RuntimeError("db password: hunter2")

        monkeypatch.setattr(
            "trl_intelligence.api.service.AssessmentService.assess_upload", boom
        )
        response = client.post(
            "/api/v1/assess",
            files={"file": ("a.md", b"content content", "text/markdown")},
        )
        assert response.status_code == 500
        assert response.json()["code"] == "internal_error"
        assert "hunter2" not in response.text
        assert "Traceback" not in response.text


class TestNoArbitraryReads:
    def test_static_filesystem_paths_are_not_served(self, client: TestClient):
        for probe in ("/etc/passwd", "/.env", "/../../etc/passwd", "/api/v1/assess"):
            response = client.get(probe)
            assert response.status_code in (404, 405), probe
            assert "root:" not in response.text

    def test_assess_ignores_injected_config_paths(self, client: TestClient):
        response = client.post(
            "/api/v1/assess",
            data={"criteria_path": "/etc/passwd", "config_path": "/etc/passwd"},
            files={
                "file": (
                    "report.md",
                    (CORPUS / "03_relevant_environment.md").read_bytes(),
                    "text/markdown",
                )
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "assessed"
        assert body["criteria_set_id"] == "hackathon_trl_v1"


def test_repo_contains_no_subprocess_shell_flags_in_api_layer():
    """Static guard: the API layer must not use shell=True or os.system."""
    api_dir = Path(__file__).resolve().parents[1] / "src" / "trl_intelligence" / "api"
    for source in api_dir.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "shell=True" not in text, source.name
        assert "os.system" not in text, source.name
        assert "os.popen" not in text, source.name
        assert "eval(" not in text, source.name
        assert "exec(" not in text, source.name
