#!/usr/bin/env python3
"""Offline TRL assessment: load a document, extract evidence, print a JSON result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trl_intelligence.pipeline import AssessmentRequest, assess_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", default=None, help="path to source document")
    parser.add_argument("--text", default=None, help="inline document text")
    parser.add_argument("--config", default=None, help="pipeline config path")
    parser.add_argument("--criteria", default=None, help="criteria config path")
    parser.add_argument("--signal", default=None, help="joblib model for a baseline signal")
    parser.add_argument(
        "--signal-variant",
        default="full",
        choices=["full", "masked"],
        help="text variant for the baseline signal",
    )
    args = parser.parse_args()

    if args.doc is None and args.text is None:
        parser.error("provide --doc or --text")

    request = AssessmentRequest(
        path=Path(args.doc) if args.doc else None,
        text=args.text,
        source_type="txt" if args.text is not None else None,
        config_path=Path(args.config) if args.config else None,
        criteria_path=Path(args.criteria) if args.criteria else None,
        signal_model_path=Path(args.signal) if args.signal else None,
        signal_variant=args.signal_variant,
    )
    result = assess_document(request)
    print(result.to_json())
    return 0 if result.status in ("assessed", "insufficient_evidence") else 2


if __name__ == "__main__":
    raise SystemExit(main())
