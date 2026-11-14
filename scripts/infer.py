#!/usr/bin/env python3
"""Offline inference: load a serialized model and predict TRL from text/file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trl_intelligence.prediction.artifacts import predict_texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to joblib model")
    parser.add_argument("--text", default=None, help="single text string")
    parser.add_argument("--file", default=None, help="file with one document per line")
    args = parser.parse_args()

    if args.text is not None:
        texts = [args.text]
    elif args.file is not None:
        texts = [
            line.strip()
            for line in Path(args.file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        parser.error("provide --text or --file")

    preds = predict_texts(args.model, texts)
    print(json.dumps({"n": len(preds), "predictions": [int(p) for p in preds]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
