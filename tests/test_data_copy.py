"""Executable verification of the Option A raw-data copy and its provenance.

Skipped when data/raw/techport/pages/ is absent (raw data is gitignored and
must never be committed, so fresh clones will not have it).
"""

import json
from pathlib import Path

import pytest

TECHPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "techport"
PAGES_DIR = TECHPORT_DIR / "pages"
SOURCE_MD = TECHPORT_DIR / "SOURCE.md"

pytestmark = pytest.mark.skipif(
    not PAGES_DIR.is_dir(),
    reason="raw TechPort pages not present (gitignored)",
)


def _page_paths() -> list[Path]:
    return sorted(PAGES_DIR.glob("offset_*.json"))


def test_source_md_documents_provenance():
    assert SOURCE_MD.is_file()
    text = SOURCE_MD.read_text(encoding="utf-8")
    assert "techport.nasa.gov" in text
    assert "21161" in text.replace(",", "")
    assert "2026-11-02" in text


def test_forty_three_page_files():
    assert len(_page_paths()) == 43


def test_offsets_are_contiguous():
    offsets = [int(p.stem.split("_")[1]) for p in _page_paths()]
    assert offsets == list(range(0, 21500, 500))


def test_record_count_and_unique_project_ids():
    total = 0
    ids: set[int] = set()
    for path in _page_paths():
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results") or []
        total += len(results)
        ids.update(rec["projectId"] for rec in results)
    assert total == 21161
    assert len(ids) == 21161
