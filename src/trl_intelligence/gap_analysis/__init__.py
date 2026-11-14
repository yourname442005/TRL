"""Gap analysis: what is demonstrated, what is missing, and what to do next."""

from trl_intelligence.assessment.models import (
    DemonstratedCriterion,
    GapAnalysis,
    GapItem,
    Recommendation,
)
from trl_intelligence.gap_analysis.analyze import analyze

__all__ = [
    "DemonstratedCriterion",
    "GapAnalysis",
    "GapItem",
    "Recommendation",
    "analyze",
]
