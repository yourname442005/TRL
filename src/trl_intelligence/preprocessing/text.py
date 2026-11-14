"""Text cleaning and TRL leakage masking for TechPort descriptions."""

from __future__ import annotations

import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TRL_RE = re.compile(
    r"\bTRL\s*[1-9]\b|\bTRL\b|technology\s+readiness\s+level",
    re.IGNORECASE,
)


def clean_text(value: str | None) -> str:
    """Strip tags, unescape entities, collapse whitespace."""
    if not value:
        return ""
    text = _TAG_RE.sub(" ", value)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def mask_trl(text: str | None) -> str:
    """Replace TRL mentions with [MASKED] to reduce label leakage."""
    if not text:
        return ""
    return _TRL_RE.sub("[MASKED]", text)
