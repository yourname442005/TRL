"""Polarity and confidence scoring for criterion matches (policy v1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trl_intelligence.extraction.match import Match, hint_hits, token_sequence

CONFIDENCE_BASES = {"supports": 0.7, "insufficient": 0.4, "contradicts": 0.6}
DEFAULT_EVIDENCE_TYPE = "mention"
EXTRA_HINT_CONFIDENCE = 0.05


@dataclass(frozen=True)
class ScoredMatch:
    match: Match
    polarity: str
    evidence_type: str
    strength: str
    confidence: float


def _cue_hit(cues: Sequence[str], tokens: list[str]) -> bool:
    return any(hint_hits(cue, tokens, guard_nominal=True) for cue in cues)


def infer_evidence_type(text: str, config: Mapping[str, Any]) -> str:
    tokens = token_sequence(text)
    for evidence_type, cues in config["extraction"]["type_cues"].items():
        if any(hint_hits(cue, tokens) for cue in cues):
            return evidence_type
    return DEFAULT_EVIDENCE_TYPE


def score(match: Match, config: Mapping[str, Any]) -> ScoredMatch:
    """Score polarity by cue precedence, then type, strength, and confidence."""
    tokens = token_sequence(match.span.text)
    extraction = config["extraction"]
    if _cue_hit(extraction["negation_cues"], tokens):
        polarity = "contradicts"
    elif _cue_hit(extraction["prospective_cues"], tokens):
        polarity = "insufficient"
    elif _cue_hit(extraction["assertion_cues"], tokens):
        polarity = "supports"
    else:
        polarity = "insufficient"
    evidence_type = infer_evidence_type(match.span.text, config)
    strength = config["strength_map"][evidence_type]
    extra_hint_hits = max(0, len(match.hit_hints) - 1)
    confidence = CONFIDENCE_BASES[polarity] + EXTRA_HINT_CONFIDENCE * extra_hint_hits
    confidence = min(1.0, max(0.0, confidence))
    return ScoredMatch(
        match=match,
        polarity=polarity,
        evidence_type=evidence_type,
        strength=strength,
        confidence=confidence,
    )
