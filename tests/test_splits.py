"""RED tests for group-aware train/val/test splits."""

import pandas as pd
import pytest

from trl_intelligence.evaluation.splits import (
    DEFAULT_RATIOS,
    class_distribution,
    missing_classes,
    split_groups,
)


def _frame(n_groups: int = 20, rows_per_group: int = 3, labels: list[int] | None = None) -> pd.DataFrame:
    rows = []
    for g in range(n_groups):
        label = labels[g] if labels else (g % 9) + 1
        for i in range(rows_per_group):
            rows.append(
                {
                    "project_id": g * 100 + i,
                    "group_id": f"g{g}",
                    "trl": label,
                    "text": f"t{g}-{i}",
                }
            )
    return pd.DataFrame(rows)


class TestSplitGroups:
    def test_default_ratios_are_seventy_fifteen_fifteen(self):
        assert DEFAULT_RATIOS == {"train": 0.70, "val": 0.15, "test": 0.15}

    def test_ratio_approximately_70_15_15(self):
        df = _frame(n_groups=100)
        parts = split_groups(df, seed=42)
        sizes = {k: len(v["group_id"].unique()) for k, v in parts.items()}
        total = sum(sizes.values())
        assert sizes["train"] / total == pytest.approx(0.70, abs=0.05)
        assert sizes["val"] / total == pytest.approx(0.15, abs=0.05)
        assert sizes["test"] / total == pytest.approx(0.15, abs=0.05)

    def test_no_group_overlap(self):
        df = _frame(n_groups=50)
        parts = split_groups(df, seed=1)
        tr = set(parts["train"]["group_id"])
        va = set(parts["val"]["group_id"])
        te = set(parts["test"]["group_id"])
        assert not (tr & va)
        assert not (tr & te)
        assert not (va & te)

    def test_no_project_overlap(self):
        df = _frame(n_groups=50)
        parts = split_groups(df, seed=1)
        ids = [set(p["project_id"]) for p in parts.values()]
        assert not (ids[0] & ids[1])
        assert not (ids[0] & ids[2])
        assert not (ids[1] & ids[2])

    def test_deterministic_with_seed(self):
        df = _frame(n_groups=40)
        a = split_groups(df, seed=42)
        b = split_groups(df, seed=42)
        for key in ("train", "val", "test"):
            assert list(a[key]["project_id"]) == list(b[key]["project_id"])

    def test_all_rows_preserved(self):
        df = _frame(n_groups=30)
        parts = split_groups(df, seed=7)
        total = sum(len(p) for p in parts.values())
        assert total == len(df)


class TestClassDistribution:
    def test_counts_by_split(self):
        df = _frame(n_groups=30, labels=[1, 2, 3] * 10)
        parts = split_groups(df, seed=42)
        dist = class_distribution(parts["train"])
        assert set(dist) <= {1, 2, 3}
        assert sum(dist.values()) == len(parts["train"])

    def test_missing_classes_report(self):
        dist = {1: 10, 2: 5}
        assert missing_classes(dist, all_classes=range(1, 10)) == [3, 4, 5, 6, 7, 8, 9]

    def test_no_missing_when_full(self):
        dist = {k: 2 for k in range(1, 10)}
        assert missing_classes(dist, all_classes=range(1, 10)) == []
