"""Deterministic extractor: composes segmentation, detection, match, score, records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from trl_intelligence.evidence.models import EvidenceSet
from trl_intelligence.extraction.candidates import segment
from trl_intelligence.extraction.detect import detect
from trl_intelligence.extraction.match import match_candidate
from trl_intelligence.extraction.records import (
    EXTRACTOR_VERSION,
    build_evidence_item,
)
from trl_intelligence.extraction.score import score

if TYPE_CHECKING:
    from trl_intelligence.assessment.criteria import CriteriaRegistry
    from trl_intelligence.document.models import Document


class DeterministicExtractor:
    """Offline extractor implementing the EvidenceExtractor protocol."""

    version = EXTRACTOR_VERSION

    def extract(
        self,
        document: Document,
        registry: CriteriaRegistry,
        config: Mapping[str, Any],
    ) -> EvidenceSet:
        candidates = detect(segment(document, config), config)
        items = []
        for candidate in candidates:
            for match in match_candidate(candidate, registry):
                scored = score(match, config)
                items.append(build_evidence_item(match, scored, document))
        return EvidenceSet(
            document_id=document.document_id,
            items=items,
            extraction_config=dict(config),
            criteria_set_id=registry.criteria_set_id,
            warnings=[],
        )
