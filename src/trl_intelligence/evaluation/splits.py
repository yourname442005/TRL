"""Group-aware train/val/test splitting for TechPort rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

DEFAULT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}


def split_groups(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: Mapping[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Split rows into train/val/test so no group_id crosses partitions.

    Groups are shuffled with ``seed`` then allocated by cumulative ratio.
    """
    ratios = dict(ratios or DEFAULT_RATIOS)
    if set(ratios) != {"train", "val", "test"}:
        raise ValueError("ratios must include train, val, and test")
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("ratios must sum to a positive value")

    if "group_id" not in df.columns:
        raise ValueError("df must contain group_id")

    groups = df["group_id"].drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = round(n * ratios["train"] / total_ratio)
    n_val = round(n * ratios["val"] / total_ratio)
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    assignment = {
        "train": set(groups[:n_train]),
        "val": set(groups[n_train : n_train + n_val]),
        "test": set(groups[n_train + n_val :]),
    }
    return {
        name: df[df["group_id"].isin(gids)].reset_index(drop=True)
        for name, gids in assignment.items()
    }


def class_distribution(frame: pd.DataFrame, label_col: str = "trl") -> dict[int, int]:
    """Count rows per label in a frame."""
    counts = frame[label_col].value_counts()
    return {int(k): int(v) for k, v in sorted(counts.items())}


def missing_classes(
    dist: Mapping[int, int],
    all_classes: Iterable[int] = range(1, 10),
) -> list[int]:
    """Return classes with zero rows in ``dist``."""
    present = {int(k) for k, v in dist.items() if v > 0}
    return [int(c) for c in all_classes if int(c) not in present]
