"""Normalized document representation for evidence-traceable TRL assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BLOCK_KINDS = ("heading", "paragraph", "list_item", "code", "table_caption")
SOURCE_TYPES = ("pdf", "txt", "md")


class DocumentLoadError(Exception):
    """Raised when a source file cannot be loaded into a Document."""

    def __init__(self, reason: str, source_path: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.source_path = source_path


class UnsupportedFormatError(DocumentLoadError):
    """Raised when the file suffix is not a supported source type."""


@dataclass(frozen=True)
class TextBlock:
    block_id: str
    kind: str
    text: str
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        if self.kind not in BLOCK_KINDS:
            raise ValueError(f"unknown block kind: {self.kind}")
        if self.start_char < 0 or self.end_char < self.start_char:
            raise ValueError("block offsets must satisfy 0 <= start <= end")


@dataclass(frozen=True)
class Page:
    page_number: int
    text: str
    blocks: list[TextBlock] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page numbers are 1-based")


@dataclass(frozen=True)
class Section:
    section_id: str
    heading: str
    level: int
    path: list[str]
    page_start: int
    page_end: int
    block_ids: list[str]


@dataclass(frozen=True)
class Document:
    document_id: str
    source_path: str
    source_type: str
    title: str | None
    metadata: dict[str, Any]
    pages: list[Page]
    sections: list[Section]
    warnings: list[str]

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"unknown source type: {self.source_type}")

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages)

    def blocks_of_page(self, page_number: int) -> list[TextBlock]:
        for page in self.pages:
            if page.page_number == page_number:
                return list(page.blocks)
        raise KeyError(f"no page {page_number}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "title": self.title,
            "page_count": self.page_count,
            "metadata": dict(self.metadata),
            "sections": [
                {
                    "section_id": s.section_id,
                    "heading": s.heading,
                    "level": s.level,
                    "path": list(s.path),
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                    "block_ids": list(s.block_ids),
                }
                for s in self.sections
            ],
            "warnings": list(self.warnings),
        }
