"""Evidence extraction: segmentation, detection, matching, and scoring."""

from trl_intelligence.extraction.base import (
    EvidenceExtractor,
    PipelineConfigError,
    load_pipeline_config,
)
from trl_intelligence.extraction.candidates import (
    CandidateSpan,
    segment,
    split_sentences,
)
from trl_intelligence.extraction.detect import detect
from trl_intelligence.extraction.match import Match, match_candidate
from trl_intelligence.extraction.pipeline import DeterministicExtractor
from trl_intelligence.extraction.records import build_evidence_item
from trl_intelligence.extraction.score import ScoredMatch, score

__all__ = [
    "CandidateSpan",
    "DeterministicExtractor",
    "EvidenceExtractor",
    "Match",
    "PipelineConfigError",
    "ScoredMatch",
    "build_evidence_item",
    "detect",
    "load_pipeline_config",
    "match_candidate",
    "score",
    "segment",
    "split_sentences",
]
