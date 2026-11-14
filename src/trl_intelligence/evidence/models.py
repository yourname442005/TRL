"""Evidence schema: traceable evidence items and per-document evidence sets."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

POLARITIES = ("supports", "contradicts", "insufficient")
EVIDENCE_TYPES = (
    "mention",
    "design",
    "report",
    "analysis",
    "simulation",
    "test",
    "demonstration",
    "operation",
)
STRENGTHS = ("weak", "moderate", "strong")
QUOTE_MAX_CHARS = 300


def make_evidence_id(
    *,
    document_id: str,
    page_number: int | None,
    block_id: str,
    start_char: int,
    criterion_id: str,
) -> str:
    payload = f"{document_id}:{page_number}:{block_id}:{start_char}:{criterion_id}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class EvidenceItem:
    document_id: str
    page_number: int | None
    section_heading: str | None
    block_id: str
    start_char: int
    end_char: int
    quote: str
    normalized: str
    criterion_id: str
    trl_level: int
    evidence_type: str
    polarity: str
    strength: str
    confidence: float
    extraction_method: str
    extractor_version: str
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not 1 <= self.trl_level <= 9:
            raise ValueError(f"trl_level must be in 1..9, got {self.trl_level}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.polarity not in POLARITIES:
            raise ValueError(f"unknown polarity: {self.polarity}")
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError(f"unknown evidence type: {self.evidence_type}")
        if self.strength not in STRENGTHS:
            raise ValueError(f"unknown strength: {self.strength}")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page numbers are 1-based")
        if len(self.quote) > QUOTE_MAX_CHARS:
            clipped = self.quote[: QUOTE_MAX_CHARS - 1] + "…"
            object.__setattr__(self, "quote", clipped)
        object.__setattr__(
            self,
            "evidence_id",
            make_evidence_id(
                document_id=self.document_id,
                page_number=self.page_number,
                block_id=self.block_id,
                start_char=self.start_char,
                criterion_id=self.criterion_id,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceSet:
    document_id: str
    items: list[EvidenceItem] = field(default_factory=list)
    extraction_config: dict[str, Any] = field(default_factory=dict)
    criteria_set_id: str = ""
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sort()

    @staticmethod
    def _sort_key(item: EvidenceItem) -> tuple[int, str, int, int]:
        return (
            item.trl_level,
            item.criterion_id,
            item.page_number if item.page_number is not None else 0,
            item.start_char,
        )

    def add(self, item: EvidenceItem) -> None:
        self.items.append(item)
        self.sort()

    def sort(self) -> None:
        self.items.sort(key=self._sort_key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "items": [item.to_dict() for item in self.items],
            "extraction_config": self.extraction_config,
            "criteria_set_id": self.criteria_set_id,
            "warnings": list(self.warnings),
        }
