"""Load and filter NASA TechPort search API page dumps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from trl_intelligence.preprocessing.text import clean_text


def load_page_records(pages_dir: str | Path) -> list[dict]:
    """Read all offset_*.json pages and return concatenated results."""
    root = Path(pages_dir)
    records: list[dict] = []
    for path in sorted(root.glob("offset_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload.get("results") or [])
    return records


def _is_valid_trl(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 9


def load_usable_dataset(pages_dir: str | Path, min_chars: int = 200) -> pd.DataFrame:
    """Return usable rows: valid trlCurrent 1-9 and cleaned description >= min_chars.

    group_id is sha1 of cleaned text so exact duplicates stay in one split group.
    """
    rows: list[dict] = []
    for rec in load_page_records(pages_dir):
        trl = rec.get("trlCurrent")
        if not _is_valid_trl(trl):
            continue
        text = clean_text(rec.get("description"))
        if len(text) < min_chars:
            continue
        program = rec.get("program") or {}
        rows.append(
            {
                "project_id": int(rec["projectId"]),
                "title": clean_text(rec.get("title")),
                "text": text,
                "trl": int(trl),
                "group_id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
                "status": rec.get("status") or "",
                "program": program.get("acronym") or "",
            }
        )
    return pd.DataFrame(
        rows,
        columns=["project_id", "title", "text", "trl", "group_id", "status", "program"],
    )
