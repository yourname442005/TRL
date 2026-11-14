#!/usr/bin/env python3
"""Measure end-to-end stage timings for the Milestone 3 validation report.

Runs the pipeline stages directly (load -> normalize -> extract -> aggregate ->
analyze) so per-stage costs are visible. Measurement only; no optimization.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pymupdf

from trl_intelligence.assessment.aggregate import aggregate
from trl_intelligence.assessment.criteria import CriteriaRegistry
from trl_intelligence.document.loaders import document_from_text, load_document
from trl_intelligence.document.normalize import normalize
from trl_intelligence.extraction.base import load_pipeline_config
from trl_intelligence.extraction.pipeline import DeterministicExtractor
from trl_intelligence.gap_analysis.analyze import analyze

CONFIG = load_pipeline_config(ROOT / "configs" / "pipeline_v1.json")
REGISTRY = CriteriaRegistry.from_file(ROOT / "configs" / "trl_criteria_v1.json")
EXTRACTOR = DeterministicExtractor()

BODY = (
    "Component validation of the monitoring breadboard was verified in lab testing, "
    "and the results were recorded in the acceptance report. "
    "The integration risk assessment was validated against the interface test plan, "
    "and the findings were recorded. "
    "The component test was executed in a relevant environment, and the test results "
    "show full functionality. "
    "The representative environment fidelity was justified in testing against the "
    "service profile, and the results were recorded. "
)


def make_pdf(page_count: int, path: Path) -> Path:
    doc = pymupdf.open()
    for index in range(page_count):
        page = doc.new_page()
        page.insert_text((72, 50), "PERF FIXTURE HEADER", fontsize=8)
        page.insert_text((72, 100), f"SECTION {index + 1}", fontsize=12)
        page.insert_textbox(pymupdf.Rect(72, 130, 540, 760), "\n\n".join([BODY] * 4), fontsize=9)
        page.insert_text((300, 800), str(index + 1), fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


def measure(label: str, source) -> dict:
    t0 = time.perf_counter()
    document = source()
    t1 = time.perf_counter()
    document = normalize(document)
    t2 = time.perf_counter()
    evidence_set = EXTRACTOR.extract(document, REGISTRY, CONFIG)
    t3 = time.perf_counter()
    assessment = aggregate(REGISTRY, evidence_set, CONFIG, signals=[])
    t4 = time.perf_counter()
    analyze(assessment, REGISTRY, evidence_set)
    t5 = time.perf_counter()
    return {
        "case": label,
        "load_ms": round((t1 - t0) * 1000, 1),
        "normalize_ms": round((t2 - t1) * 1000, 1),
        "extract_ms": round((t3 - t2) * 1000, 1),
        "aggregate_ms": round((t4 - t3) * 1000, 1),
        "gaps_ms": round((t5 - t4) * 1000, 1),
        "total_ms": round((t5 - t0) * 1000, 1),
        "evidence": len(evidence_set.items),
        "predicted_trl": assessment.predicted_trl,
    }


def main() -> None:
    rows = []
    rows.append(measure("short text (~1.5 KB)", lambda: document_from_text(BODY * 2)))
    with tempfile.TemporaryDirectory() as tmp:
        for pages in (5, 20, 50):
            pdf = make_pdf(pages, Path(tmp) / f"{pages}_pages.pdf")
            rows.append(measure(f"{pages}-page PDF", lambda p=pdf: load_document(p)))
    header = (
        f"{'case':18s} {'load':>9s} {'normalize':>10s} {'extract':>9s} "
        f"{'aggregate':>10s} {'gaps':>7s} {'total':>9s} {'evid':>5s} {'trl':>4s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['case']:18s} {row['load_ms']:8.1f}m {row['normalize_ms']:9.1f}m "
            f"{row['extract_ms']:8.1f}m {row['aggregate_ms']:9.1f}m "
            f"{row['gaps_ms']:6.1f}m {row['total_ms']:8.1f}m "
            f"{row['evidence']:5d} {row['predicted_trl']!s:>4s}"
        )


if __name__ == "__main__":
    main()
