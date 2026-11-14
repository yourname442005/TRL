"""Thin assessment service: input validation, staging, pipeline call, response shaping.

This module is an adapter around the existing pure-Python pipeline. It contains
no assessment logic: every decision about TRL, evidence, and gaps comes from
``assess_document`` exactly as the CLI uses it.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from trl_intelligence.extraction.records import EXTRACTOR_VERSION
from trl_intelligence.pipeline import AssessmentRequest, assess_document

ALLOWED_EXTENSIONS = frozenset({".pdf", ".txt", ".text", ".md", ".markdown"})
ALLOWED_CONTENT_TYPES = frozenset(
    {
        "",
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
        "text/plain",
        "text/markdown",
        "text/x-markdown",
    }
)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_TEMP_PREFIX = "trl-assess-"
_PDF_MAGIC = b"%PDF"
_MAX_SOURCE_NAME = 255


class ServiceValidationError(Exception):
    """Request-level input problem; never surfaces as a traceback."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AssessmentService:
    """Independently testable facade: bytes/filename -> API response payload."""

    def assess_upload(
        self,
        data: bytes,
        filename: str,
        content_type: str | None,
    ) -> dict[str, Any]:
        source_name = self._sanitize_filename(filename)
        extension = Path(source_name).suffix.lower()
        self._validate(data, extension, content_type, source_name)

        temp_dir = tempfile.mkdtemp(prefix=_TEMP_PREFIX)
        try:
            staged = Path(temp_dir) / f"upload{extension}"
            staged.write_bytes(data)
            started = time.perf_counter()
            result = assess_document(AssessmentRequest(path=staged))
            elapsed_ms = (time.perf_counter() - started) * 1000
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return self._wrap(result, source_name=source_name, byte_size=len(data), elapsed_ms=elapsed_ms)

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        name = Path(str(filename).replace("\\", "/")).name
        name = "".join(char for char in name if char.isprintable())
        return name[:_MAX_SOURCE_NAME] or "upload"

    @staticmethod
    def _validate(
        data: bytes,
        extension: str,
        content_type: str | None,
        source_name: str,
    ) -> None:
        if not data:
            raise ServiceValidationError("empty_file", "uploaded file is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ServiceValidationError(
                "file_too_large",
                f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
            )
        if extension not in ALLOWED_EXTENSIONS:
            raise ServiceValidationError(
                "unsupported_type",
                f"unsupported file extension for {source_name!r}; "
                f"allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )
        normalized_type = (content_type or "").split(";")[0].strip().lower()
        if normalized_type not in ALLOWED_CONTENT_TYPES:
            raise ServiceValidationError(
                "unsupported_content_type",
                f"unsupported content type: {normalized_type or '(none)'}",
            )
        if extension == ".pdf" and not data.startswith(_PDF_MAGIC):
            raise ServiceValidationError(
                "invalid_content",
                "file extension is .pdf but content does not start with %PDF",
            )

    @staticmethod
    def _wrap(result: Any, *, source_name: str, byte_size: int, elapsed_ms: float) -> dict[str, Any]:
        payload = result.to_dict()
        document = result.document
        if document is not None:
            payload["document"] = {
                "source_name": source_name,
                "source_type": document.source_type,
                "document_id": document.document_id,
                "title": document.title,
                "page_count": document.metadata.get("page_count"),
                "byte_size": document.metadata.get("byte_size"),
                "loader": document.metadata.get("loader"),
                "warnings": list(document.warnings),
            }
        else:
            payload["document"] = {
                "source_name": source_name,
                "source_type": None,
                "document_id": None,
                "title": None,
                "page_count": None,
                "byte_size": byte_size,
                "loader": None,
                "warnings": [],
            }
        payload["processing"] = {
            "processing_time_ms": round(elapsed_ms, 1),
            "extractor_version": EXTRACTOR_VERSION,
        }
        return payload
