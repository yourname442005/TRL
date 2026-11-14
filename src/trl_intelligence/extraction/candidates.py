"""Deterministic candidate segmentation: paragraphs to sentence spans."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from trl_intelligence.document.models import Document

_ABBREVIATION_TOKENS = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "etc",
    "e.g",
    "i.e",
    "vs",
    "cf",
    "fig",
    "no",
    "approx",
    "jr",
    "sr",
    "st",
    "vol",
    "inc",
    "ltd",
    "co",
    "corp",
    "est",
    "dept",
    "univ",
    "a.m",
    "p.m",
    "u.s",
}
_TERMINATORS = ".!?"
_TOKEN_RE = re.compile(r"[A-Za-z0-9.]")


@dataclass(frozen=True)
class CandidateSpan:
    text: str
    page: int
    block_id: str
    section: str
    start: int
    end: int


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split text into (start, end, sentence) spans, abbreviation-safe."""
    if not text.strip():
        return []
    n = len(text)
    boundaries: list[int] = []
    i = 0
    while i < n:
        if text[i] not in _TERMINATORS:
            i += 1
            continue
        j = i
        while j < n and text[j] in _TERMINATORS:
            j += 1
        k = i
        while k > 0 and _TOKEN_RE.match(text[k - 1]):
            k -= 1
        token = text[k:i].lower()
        m = j
        while m < n and text[m].isspace():
            m += 1
        next_is_boundary = m >= n or text[m].isupper() or text[m].isdigit() or text[m] in "\"'("
        if token not in _ABBREVIATION_TOKENS and next_is_boundary:
            boundaries.append(j)
        i = j
    sentences: list[tuple[int, int, str]] = []
    prev = 0
    for end in boundaries:
        start = prev
        while start < end and text[start].isspace():
            start += 1
        seg_end = end
        while seg_end > start and text[seg_end - 1].isspace():
            seg_end -= 1
        if seg_end > start:
            sentences.append((start, seg_end, text[start:seg_end]))
        prev = end
    tail_start = prev
    while tail_start < n and text[tail_start].isspace():
        tail_start += 1
    if tail_start < n:
        tail_end = n
        while tail_end > tail_start and text[tail_end - 1].isspace():
            tail_end -= 1
        sentences.append((tail_start, tail_end, text[tail_start:tail_end]))
    return sentences


def segment(document: Document, config: Mapping[str, Any]) -> list[CandidateSpan]:
    """Segment a document's paragraph blocks into capped sentence candidates."""
    extraction = config["extraction"]
    min_chars = int(extraction["min_candidate_chars"])
    max_per_section = int(extraction["max_candidates_per_section"])
    blocks_by_id = {block.block_id: block for page in document.pages for block in page.blocks}
    page_by_block = {
        block.block_id: page.page_number for page in document.pages for block in page.blocks
    }
    results: list[CandidateSpan] = []
    for section in document.sections:
        kept = 0
        for block_id in section.block_ids:
            if kept >= max_per_section:
                break
            block = blocks_by_id.get(block_id)
            if block is None or block.kind != "paragraph":
                continue
            for start, end, sentence in split_sentences(block.text):
                if kept >= max_per_section:
                    break
                if len(sentence) < min_chars:
                    continue
                results.append(
                    CandidateSpan(
                        text=sentence,
                        page=page_by_block[block_id],
                        block_id=block_id,
                        section=section.heading,
                        start=start,
                        end=end,
                    )
                )
                kept += 1
    return results
