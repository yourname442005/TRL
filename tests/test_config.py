"""Tests pinning baseline_v1 config to the approved Milestone 1 decisions."""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "baseline_v1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class TestBaselineConfig:
    def test_seed_is_forty_two(self):
        assert _config()["seed"] == 42

    def test_split_ratios_are_group_aware_70_15_15(self):
        assert _config()["split_ratios"] == {"train": 0.7, "val": 0.15, "test": 0.15}

    def test_primary_variant_is_masked(self):
        assert _config()["primary_variant"] == "masked"

    def test_both_text_variants_enabled(self):
        assert _config()["text_variants"] == ["full", "masked"]

    def test_three_baseline_models(self):
        assert _config()["models"] == ["b1", "b2", "b3"]
