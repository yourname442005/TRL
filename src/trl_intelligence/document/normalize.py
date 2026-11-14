"""Document normalization: repeated headers/footers, page numbers, sections."""

from __future__ import annotations

import math
import re
from dataclasses import replace

from trl_intelligence.document.loaders import _markdown_blocks, _paragraph_blocks
from trl_intelligence.document.models import Document, Page, Section, TextBlock

_PAGE_NUM_RE = re.compile(r"\d{1,4}|[ivxlc]{1,6}", re.IGNORECASE)
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize(document: Document) -> Document:
    """Strip repeated headers/footers and page-number lines, rebuild blocks, add sections."""
    metadata = dict(document.metadata)
    texts = [page.text for page in document.pages]

    texts, removed_numbers = _strip_page_number_lines(texts)
    texts, removed_headers = _strip_repeated_lines(texts)

    pages = [
        _rebuild_page(page_number=i + 1, text=texts[i], source_type=document.source_type)
        for i in range(len(document.pages))
    ]
    sections = _build_sections(pages)

    metadata["removed_page_number_lines"] = removed_numbers
    metadata["removed_repeated_lines"] = removed_headers
    return Document(
        document_id=document.document_id,
        source_path=document.source_path,
        source_type=document.source_type,
        title=document.title,
        metadata=metadata,
        pages=pages,
        sections=sections,
        warnings=list(document.warnings),
    )


def _non_empty_indices(lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if line.strip()]


def _roman_to_int(text: str) -> int:
    chars = text.lower()
    total = 0
    prev = 0
    for char in reversed(chars):
        value = _ROMAN_VALUES.get(char, 0)
        total += -value if value < prev else value
        prev = max(prev, value)
    return total


def _is_page_number_line(line: str, page_number: int) -> bool:
    stripped = line.strip()
    if not _PAGE_NUM_RE.fullmatch(stripped):
        return False
    if stripped.isdigit():
        return abs(int(stripped) - page_number) <= 1
    return _roman_to_int(stripped) == page_number


def _strip_page_number_lines(texts: list[str]) -> tuple[list[str], int]:
    removed = 0
    out: list[str] = []
    for page_number, text in enumerate(texts, start=1):
        lines = text.split("\n")
        indices = _non_empty_indices(lines)
        drop: set[int] = set()
        if indices:
            for i in (indices[0], indices[-1]):
                if _is_page_number_line(lines[i], page_number):
                    drop.add(i)
        if drop:
            removed += len(drop)
            lines = [line for i, line in enumerate(lines) if i not in drop]
            lines = _trim_blank_edges(lines)
        out.append("\n".join(lines))
    return out, removed


def _normalized_line(line: str) -> str:
    return _NORMALIZE_RE.sub("", line.strip().lower())


def _strip_repeated_lines(texts: list[str]) -> tuple[list[str], int]:
    if not texts:
        return list(texts), 0
    threshold = max(3, math.ceil(0.6 * len(texts)))
    counts: dict[str, set[int]] = {}
    for page_index, text in enumerate(texts):
        lines = text.split("\n")
        indices = _non_empty_indices(lines)
        for i in (indices[0], indices[-1]) if indices else ():
            norm = _normalized_line(lines[i])
            if norm:
                counts.setdefault(norm, set()).add(page_index)
    repeated = {norm for norm, pages in counts.items() if len(pages) >= threshold}

    removed = 0
    out: list[str] = []
    for page_index, text in enumerate(texts):
        lines = text.split("\n")
        indices = _non_empty_indices(lines)
        drop: set[int] = set()
        for i in (indices[0], indices[-1]) if indices else ():
            if _normalized_line(lines[i]) in repeated:
                drop.add(i)
        if drop:
            removed += len(drop)
            lines = [line for i, line in enumerate(lines) if i not in drop]
            lines = _trim_blank_edges(lines)
        out.append("\n".join(lines))
    return out, removed


def _trim_blank_edges(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _lexical_heading_level(text: str) -> int | None:
    if not text or len(text) > 80 or "\n" in text:
        return None
    if text[-1] in ".:;,":
        return None
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 3:
        return None
    if text.isupper():
        return 1
    words = text.split()
    starts_upper = [word[0].isupper() for word in words if word and word[0].isalpha()]
    if len(starts_upper) >= 2 and all(starts_upper):
        return 2
    return None


def _rebuild_page(page_number: int, text: str, source_type: str) -> Page:
    if source_type == "md":
        blocks = _markdown_blocks(text, page_number)
    else:
        blocks = []
        for block in _paragraph_blocks(text, page_number):
            level = _lexical_heading_level(block.text)
            if level is not None:
                blocks.append(replace(block, kind="heading"))
            else:
                blocks.append(block)
    return Page(page_number=page_number, text=text, blocks=blocks)


def _heading_parts(block: TextBlock) -> tuple[int, str]:
    if block.text.lstrip().startswith("#"):
        level = len(block.text.lstrip()) - len(block.text.lstrip().lstrip("#"))
        return max(level, 1), block.text.lstrip("#").strip()
    level = _lexical_heading_level(block.text) or 1
    return level, block.text.strip()


def _build_sections(pages: list[Page]) -> list[Section]:
    sections: list[dict] = []
    stack: list[tuple[int, str]] = []
    current: dict | None = None

    def open_section(level: int, name: str, page_number: int, block_id: str) -> dict:
        nonlocal current
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, name))
        current = {
            "section_id": f"s{len(sections)}",
            "heading": name,
            "level": level,
            "path": [heading for _, heading in stack],
            "page_start": page_number,
            "page_end": page_number,
            "block_ids": [block_id],
        }
        sections.append(current)
        return current

    for page in pages:
        for block in page.blocks:
            if block.kind == "heading":
                level, name = _heading_parts(block)
                open_section(level, name, page.page_number, block.block_id)
                continue
            if current is None:
                current = {
                    "section_id": f"s{len(sections)}",
                    "heading": "body",
                    "level": 0,
                    "path": ["body"],
                    "page_start": page.page_number,
                    "page_end": page.page_number,
                    "block_ids": [block.block_id],
                }
                sections.append(current)
            else:
                current["block_ids"].append(block.block_id)
                current["page_end"] = page.page_number

    return [
        Section(
            section_id=data["section_id"],
            heading=data["heading"],
            level=data["level"],
            path=data["path"],
            page_start=data["page_start"],
            page_end=data["page_end"],
            block_ids=data["block_ids"],
        )
        for data in sections
    ]
