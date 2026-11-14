"""Criterion matching: token-sequence hint hits per candidate.

A hint matches when every hint token aligns with a token of the candidate text,
allowing inflectional and morphological variants (tests/testing,
demonstrates/demonstration, operating/operational) via exact, prefix, or
shared-stem matching. Multi-token hints must align to adjacent tokens, or —
when spread apart — must include at least one morphologically related (stem)
token. Verbatim phrases therefore keep literal-phrase behavior (no bare
co-occurrence of unrelated words), while equivalent technical phrasing with a
morphological bridge is still recognized in arbitrary documents.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass

from trl_intelligence.assessment.criteria import CriteriaRegistry
from trl_intelligence.extraction.candidates import CandidateSpan

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_PREFIX_LEN = 3
_MIN_STEM_PREFIX = 6
_NOMINAL_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ance", "ence")


@dataclass(frozen=True)
class Match:
    criterion_id: str
    trl: int
    hit_hints: tuple[str, ...]
    span: CandidateSpan


def token_sequence(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def text_tokens(text: str) -> set[str]:
    return set(token_sequence(text))


def _classify(hint_token: str, text_token: str, *, guard_nominal: bool) -> str | None:
    if hint_token == text_token:
        return "exact"
    if len(hint_token) >= _MIN_PREFIX_LEN and text_token.startswith(hint_token):
        return "prefix"
    shared = 0
    for left, right in zip(hint_token, text_token):
        if left != right:
            break
        shared += 1
    if shared >= _MIN_STEM_PREFIX:
        if guard_nominal and any(
            text_token.endswith(suffix) and not hint_token.endswith(suffix)
            for suffix in _NOMINAL_SUFFIXES
        ):
            return None
        return "stem"
    return None


def hint_hits(
    hint: str, tokens: Sequence[str], *, guard_nominal: bool = False
) -> bool:
    """True when every hint token aligns with a token of the ordered text.

    Single-token hints match anywhere. Multi-token hints match when the
    aligned positions form an adjacent run, or when at least one hint token
    alignes via a morphological stem (the spread-apart equivalent-phrasing
    case).
    """
    hint_tokens = _TOKEN_RE.findall(hint.lower())
    if not hint_tokens:
        return False
    count = len(hint_tokens)
    if count == 1:
        return any(
            _classify(hint_tokens[0], token, guard_nominal=guard_nominal)
            for token in tokens
        )
    slots: list[list[int]] = []
    stem_available = False
    for hint_token in hint_tokens:
        indices: list[int] = []
        for index, token in enumerate(tokens):
            kind = _classify(hint_token, token, guard_nominal=guard_nominal)
            if kind is None:
                continue
            indices.append(index)
            if kind == "stem":
                stem_available = True
        if not indices:
            return False
        slots.append(indices)
    for start in range(len(tokens) - count + 1):
        for order in itertools.permutations(range(count)):
            if all(start + offset in slots[position] for position, offset in enumerate(order)):
                return True
    return stem_available


def match_candidate(candidate: CandidateSpan, registry: CriteriaRegistry) -> list[Match]:
    """Return one Match per criterion whose hints appear in the candidate text."""
    tokens = token_sequence(candidate.text)
    matches: list[Match] = []
    for level in registry.levels:
        for criterion in level.criteria:
            hits = tuple(hint for hint in criterion.match_hints if hint_hits(hint, tokens))
            if hits:
                matches.append(
                    Match(
                        criterion_id=criterion.criterion_id,
                        trl=level.trl,
                        hit_hints=hits,
                        span=candidate,
                    )
                )
    return matches
