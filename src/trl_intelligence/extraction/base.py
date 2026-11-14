"""Extraction protocol and pipeline configuration loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from trl_intelligence.assessment.criteria import CriteriaRegistry
    from trl_intelligence.document.models import Document
    from trl_intelligence.evidence.models import EvidenceSet

REQUIRED_PIPELINE_KEYS = (
    "schema_version",
    "extraction",
    "aggregation",
    "strength_map",
    "signal",
)


class PipelineConfigError(ValueError):
    """Raised when a pipeline configuration file is missing required sections."""


@runtime_checkable
class EvidenceExtractor(Protocol):
    def extract(
        self,
        document: Document,
        registry: CriteriaRegistry,
        config: Mapping[str, Any],
    ) -> EvidenceSet: ...


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PipelineConfigError("pipeline config must be a JSON object")
    for key in REQUIRED_PIPELINE_KEYS:
        if key not in data:
            raise PipelineConfigError(f"pipeline config missing required key: {key}")
    return data
