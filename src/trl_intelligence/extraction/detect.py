"""Technical-evidence detection: config-driven lexical cue filtering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trl_intelligence.extraction.candidates import CandidateSpan


def detect(candidates: Sequence[CandidateSpan], config: Mapping[str, Any]) -> list[CandidateSpan]:
    """Keep candidates whose text hits at least one technical cue."""
    cues = [cue.lower() for cue in config["extraction"]["technical_cues"]]
    return [
        candidate for candidate in candidates if any(cue in candidate.text.lower() for cue in cues)
    ]
