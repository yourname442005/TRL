"""Versioned TRL criteria registry loaded from a validated JSON configuration."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_STRENGTHS = ("weak", "moderate", "strong")
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
_LEVEL_SEQUENCE = tuple(range(1, 10))
_CRITERION_ID_RE = re.compile(r"T\d+\.C\d+")
_REQUIRED_CRITERION_FIELDS = (
    "criterion_id",
    "description",
    "match_hints",
    "guidance",
    "evidence_types",
    "example_artifacts",
    "min_strength",
    "weight",
)


class CriteriaValidationError(Exception):
    """Raised when a criteria configuration file fails validation."""


@dataclass(frozen=True)
class Criterion:
    criterion_id: str
    trl: int
    description: str
    match_hints: tuple[str, ...]
    guidance: str
    evidence_types: tuple[str, ...]
    example_artifacts: tuple[str, ...]
    min_strength: str
    weight: float


@dataclass(frozen=True)
class TRLLevel:
    trl: int
    name: str
    criteria: tuple[Criterion, ...]


@dataclass(frozen=True)
class CriteriaRegistry:
    criteria_set_id: str
    schema_version: str
    source_note: str
    strength_map: dict[str, str]
    levels: tuple[TRLLevel, ...]
    content_hash: str

    @classmethod
    def from_file(cls, path: str | Path) -> CriteriaRegistry:
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CriteriaRegistry:
        for key in ("criteria_set_id", "schema_version", "source_note", "strength_map", "levels"):
            if key not in data:
                raise CriteriaValidationError(f"missing required key: {key}")
        strength_map = _validate_strength_map(data["strength_map"])
        levels = _validate_levels(data["levels"])
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        content_hash = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
        return cls(
            criteria_set_id=data["criteria_set_id"],
            schema_version=data["schema_version"],
            source_note=data["source_note"],
            strength_map=strength_map,
            levels=levels,
            content_hash=content_hash,
        )

    def level(self, trl: int) -> TRLLevel:
        for level in self.levels:
            if level.trl == trl:
                return level
        raise KeyError(f"unknown TRL level: {trl}")

    def criterion(self, criterion_id: str) -> Criterion:
        for level in self.levels:
            for criterion in level.criteria:
                if criterion.criterion_id == criterion_id:
                    return criterion
        raise KeyError(f"unknown criterion_id: {criterion_id}")

    def all_criteria(self) -> list[Criterion]:
        return [criterion for level in self.levels for criterion in level.criteria]


def _validate_strength_map(strength_map: Any) -> dict[str, str]:
    if not isinstance(strength_map, dict):
        raise CriteriaValidationError("strength_map must be an object")
    if set(strength_map) != set(EVIDENCE_TYPES):
        raise CriteriaValidationError(
            "strength_map must cover exactly the evidence types: " + ", ".join(EVIDENCE_TYPES)
        )
    for evidence_type, strength in strength_map.items():
        if strength not in VALID_STRENGTHS:
            raise CriteriaValidationError(
                f"strength_map[{evidence_type!r}] must be one of {VALID_STRENGTHS}"
            )
    if strength_map["mention"] != "weak":
        raise CriteriaValidationError("strength_map rule: mention must map to weak")
    return dict(strength_map)


def _validate_levels(levels: Any) -> tuple[TRLLevel, ...]:
    if not isinstance(levels, list):
        raise CriteriaValidationError("levels must be a list")
    trl_sequence = [level.get("trl") for level in levels if isinstance(level, dict)]
    if trl_sequence != list(_LEVEL_SEQUENCE):
        raise CriteriaValidationError("levels must cover TRL 1 through 9 exactly once, in order")
    seen_ids: set[str] = set()
    validated: list[TRLLevel] = []
    for level in levels:
        trl = level["trl"]
        name = level.get("name")
        if not isinstance(name, str) or not name.strip():
            raise CriteriaValidationError(f"level {trl} must have a non-empty name")
        raw_criteria = level.get("criteria")
        if not isinstance(raw_criteria, list) or not raw_criteria:
            raise CriteriaValidationError(f"level {trl} must have at least one criterion")
        criteria: list[Criterion] = []
        for raw in raw_criteria:
            criterion = _validate_criterion(raw, trl, seen_ids)
            criteria.append(criterion)
        validated.append(TRLLevel(trl=trl, name=name, criteria=tuple(criteria)))
    return tuple(validated)


def _validate_criterion(raw: Any, trl: int, seen_ids: set[str]) -> Criterion:
    if not isinstance(raw, dict):
        raise CriteriaValidationError("each criterion must be an object")
    for field_name in _REQUIRED_CRITERION_FIELDS:
        if field_name not in raw:
            raise CriteriaValidationError(
                f"level {trl} criterion missing required field: {field_name}"
            )
    criterion_id = raw["criterion_id"]
    if not isinstance(criterion_id, str) or not _CRITERION_ID_RE.fullmatch(criterion_id):
        raise CriteriaValidationError(f"malformed criterion_id: {criterion_id!r}")
    if criterion_id in seen_ids:
        raise CriteriaValidationError(f"duplicate criterion_id: {criterion_id}")
    seen_ids.add(criterion_id)
    if not re.fullmatch(rf"T{trl}\.C\d+", criterion_id):
        raise CriteriaValidationError(
            f"criterion_id {criterion_id} does not match its level T{trl}"
        )
    description = raw["description"]
    guidance = raw["guidance"]
    if not isinstance(description, str) or not description.strip():
        raise CriteriaValidationError(f"criterion_id {criterion_id}: empty description")
    if not isinstance(guidance, str) or not guidance.strip():
        raise CriteriaValidationError(f"criterion_id {criterion_id}: empty guidance")
    match_hints = raw["match_hints"]
    if (
        not isinstance(match_hints, list)
        or not match_hints
        or not all(isinstance(hint, str) and hint.strip() for hint in match_hints)
    ):
        raise CriteriaValidationError(
            f"criterion_id {criterion_id}: match_hints must be a non-empty list of strings"
        )
    evidence_types = raw["evidence_types"]
    if (
        not isinstance(evidence_types, list)
        or not evidence_types
        or not all(item in EVIDENCE_TYPES for item in evidence_types)
    ):
        raise CriteriaValidationError(
            f"criterion_id {criterion_id}: evidence_types must be a non-empty subset of "
            + ", ".join(EVIDENCE_TYPES)
        )
    example_artifacts = raw["example_artifacts"]
    if (
        not isinstance(example_artifacts, list)
        or not example_artifacts
        or not all(isinstance(item, str) and item.strip() for item in example_artifacts)
    ):
        raise CriteriaValidationError(
            f"criterion_id {criterion_id}: example_artifacts must be a non-empty list"
        )
    min_strength = raw["min_strength"]
    if min_strength not in VALID_STRENGTHS:
        raise CriteriaValidationError(
            f"criterion_id {criterion_id}: min_strength must be one of {VALID_STRENGTHS}"
        )
    weight = raw["weight"]
    if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
        raise CriteriaValidationError(
            f"criterion_id {criterion_id}: weight must be a positive number"
        )
    return Criterion(
        criterion_id=criterion_id,
        trl=trl,
        description=description,
        match_hints=tuple(match_hints),
        guidance=guidance,
        evidence_types=tuple(evidence_types),
        example_artifacts=tuple(example_artifacts),
        min_strength=min_strength,
        weight=float(weight),
    )
