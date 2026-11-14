"""EvidenceItem construction from scored matches."""

from __future__ import annotations

import html
import re

from trl_intelligence.document.models import Document
from trl_intelligence.evidence.models import EvidenceItem
from trl_intelligence.extraction.match import Match
from trl_intelligence.extraction.score import ScoredMatch

EXTRACTION_METHOD = "deterministic_v1"
EXTRACTOR_VERSION = "1.0.0"


def normalize_span(text: str) -> str:
    """Whitespace-collapse and html-unescape a source span."""
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def build_evidence_item(match: Match, scored: ScoredMatch, document: Document) -> EvidenceItem:
    span = match.span
    return EvidenceItem(
        document_id=document.document_id,
        page_number=span.page,
        section_heading=span.section,
        block_id=span.block_id,
        start_char=span.start,
        end_char=span.end,
        quote=span.text,
        normalized=normalize_span(span.text),
        criterion_id=match.criterion_id,
        trl_level=match.trl,
        evidence_type=scored.evidence_type,
        polarity=scored.polarity,
        strength=scored.strength,
        confidence=scored.confidence,
        extraction_method=EXTRACTION_METHOD,
        extractor_version=EXTRACTOR_VERSION,
    )
