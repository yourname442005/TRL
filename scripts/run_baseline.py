#!/usr/bin/env python3
"""Train TRL baselines, evaluate full+masked, write metrics (test used only at end)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trl_intelligence.evaluation.metrics import (
    classification_metrics,
    relative_mae_delta,
)
from trl_intelligence.evaluation.splits import (
    class_distribution,
    missing_classes,
    split_groups,
)
from trl_intelligence.ingestion.techport import load_usable_dataset
from trl_intelligence.models.baselines import build_from_config
from trl_intelligence.prediction.artifacts import save_model
from trl_intelligence.preprocessing.text import mask_trl


def texts_for(frame, variant: str) -> list[str]:
    if variant == "full":
        return frame["text"].tolist()
    if variant == "masked":
        return [mask_trl(t) for t in frame["text"].tolist()]
    raise ValueError(variant)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/baseline_v1.json"))
    parser.add_argument("--models", default=None, help="comma subset: b1,b2,b3")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    pages_dir = ROOT / cfg["pages_dir"]
    out_dir = ROOT / cfg["output_dir"]
    model_dir = ROOT / cfg["model_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_names = args.models.split(",") if args.models else list(cfg["models"])
    seed = int(cfg["seed"])

    print(f"Loading pages from {pages_dir} ...")
    df = load_usable_dataset(pages_dir, min_chars=int(cfg["min_chars"]))
    print(f"Usable rows: {len(df)}")

    parts = split_groups(df, seed=seed, ratios=cfg["split_ratios"])
    split_report = {
        name: {
            "n_rows": len(frame),
            "n_groups": int(frame["group_id"].nunique()),
            "class_distribution": class_distribution(frame),
            "missing_classes": missing_classes(class_distribution(frame)),
        }
        for name, frame in parts.items()
    }
    (out_dir / "split_report.json").write_text(json.dumps(split_report, indent=2))
    (out_dir / "split_indices.json").write_text(
        json.dumps(
            {k: v["project_id"].astype(int).tolist() for k, v in parts.items()},
            indent=2,
        )
    )
    print("Split report:", json.dumps(split_report, indent=2))

    train, val, test = parts["train"], parts["val"], parts["test"]
    # Model selection uses train fit + val metrics only. Test evaluated once at end.
    results: dict = {"config": cfg, "splits": split_report, "models": {}}

    for name in model_names:
        results["models"][name] = {}
        for variant in cfg.get("text_variants", ["full", "masked"]):
            print(f"\n=== {name} / {variant} ===")
            model = build_from_config(name, cfg)
            model.fit(texts_for(train, variant), train["trl"].tolist())
            val_pred = model.predict(texts_for(val, variant))
            val_metrics = classification_metrics(val["trl"].tolist(), list(val_pred))
            print(
                f"VAL  mae={val_metrics['mae']:.4f} "
                f"within1={val_metrics['within_1']:.4f} "
                f"macro_f1={val_metrics['macro_f1']:.4f} "
                f"qwk={val_metrics['qwk']:.4f}"
            )
            results["models"][name][f"val_{variant}"] = val_metrics
            # persist model for this variant
            save_model(model, model_dir / f"{name}_{variant}.joblib")

        # Final test evaluation once per variant (selection already done on val)
        for variant in cfg.get("text_variants", ["full", "masked"]):
            model_path = model_dir / f"{name}_{variant}.joblib"
            from trl_intelligence.prediction.artifacts import load_model

            model = load_model(model_path)
            test_pred = model.predict(texts_for(test, variant))
            test_metrics = classification_metrics(test["trl"].tolist(), list(test_pred))
            print(
                f"TEST mae={test_metrics['mae']:.4f} "
                f"within1={test_metrics['within_1']:.4f} "
                f"macro_f1={test_metrics['macro_f1']:.4f} "
                f"qwk={test_metrics['qwk']:.4f}"
            )
            results["models"][name][f"test_{variant}"] = test_metrics

        full = results["models"][name].get("test_full")
        masked = results["models"][name].get("test_masked")
        if full and masked:
            results["models"][name]["mae_delta_full_vs_masked"] = relative_mae_delta(
                mae_full=full["mae"], mae_masked=masked["mae"]
            )

    # Primary ranking on masked test MAE
    ranking = sorted(
        (
            name
            for name in results["models"]
            if "test_masked" in results["models"][name]
        ),
        key=lambda n: results["models"][n]["test_masked"]["mae"],
    )
    results["ranking_by_masked_test_mae"] = ranking

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, allow_nan=True))
    print(f"\nWrote {metrics_path}")
    print("Ranking (masked test MAE):", ranking)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
