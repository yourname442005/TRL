"""FastAPI adapter around the assessment service.

HTTP concerns only: request parsing, validation, status codes, controlled error
bodies. All assessment logic lives in the service and the existing pipeline.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from trl_intelligence.api import service as service_module
from trl_intelligence.api.service import AssessmentService, ServiceValidationError
from trl_intelligence.assessment.criteria import CriteriaRegistry
from trl_intelligence.assessment.models import POLICY_VERSION
from trl_intelligence.pipeline import DEFAULT_CRITERIA_PATH

logger = logging.getLogger("trl_intelligence.api")

_ERROR_STATUS = {
    "empty_file": 400,
    "missing_filename": 400,
    "file_too_large": 413,
    "unsupported_type": 415,
    "unsupported_content_type": 415,
    "invalid_content": 415,
}
_NOT_FOUND_CODES = {404: "not_found", 405: "method_not_allowed"}


def _error(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"status": "failed", "code": code, "error": message},
    )


def create_app() -> FastAPI:
    app = FastAPI(title="TRL Intelligence API", version="1.0.0", redoc_url=None)
    registry = CriteriaRegistry.from_file(DEFAULT_CRITERIA_PATH)
    assessment_service = AssessmentService()
    app.state.service = assessment_service

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "trl-intelligence",
            "policy_version": POLICY_VERSION,
            "criteria_set_id": registry.criteria_set_id,
            "criteria_content_hash": registry.content_hash,
        }

    @app.post("/api/v1/assess")
    async def assess(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        max_bytes = service_module.MAX_UPLOAD_BYTES
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ServiceValidationError(
                "file_too_large",
                f"file exceeds the {max_bytes // (1024 * 1024)} MB limit",
            )
        if not file.filename:
            raise ServiceValidationError("missing_filename", "upload is missing a filename")
        return assessment_service.assess_upload(data, file.filename, file.content_type)

    @app.exception_handler(ServiceValidationError)
    async def handle_validation_error(_request, exc: ServiceValidationError) -> JSONResponse:
        return _error(exc.code, exc.message, _ERROR_STATUS.get(exc.code, 400))

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        _request, exc: RequestValidationError
    ) -> JSONResponse:
        detail = [
            {
                "loc": [str(part) for part in error.get("loc", ())],
                "msg": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "status": "failed",
                "code": "invalid_request",
                "error": "request validation failed",
                "detail": detail,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        _request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = _NOT_FOUND_CODES.get(exc.status_code, "http_error")
        message = exc.detail if isinstance(exc.detail, str) else "request could not be handled"
        return _error(code, message, exc.status_code)

    @app.exception_handler(Exception)
    async def handle_unexpected(_request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled API error: %s", exc)
        return _error("internal_error", "internal server error", 500)

    return app


app = create_app()
