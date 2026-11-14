"""RED tests for TechPort page ingestion."""

import json
from pathlib import Path

import pytest

from trl_intelligence.ingestion.techport import load_page_records, load_usable_dataset


def _write_page(dirpath: Path, offset: int, records: list[dict]) -> None:
    payload = {"offset": offset, "limit": 500, "results": records}
    (dirpath / f"offset_{offset:06d}.json").write_text(json.dumps(payload))


def _rec(pid: int, desc: str, trl: int | None = 3, title: str | None = None) -> dict:
    return {
        "projectId": pid,
        "title": title or f"Project {pid}",
        "description": desc,
        "trlCurrent": trl,
        "status": "Completed",
        "program": {"acronym": "SBIR/STTR"},
    }


@pytest.fixture
def pages_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pages"
    d.mkdir()
    long_desc = "A" * 200
    _write_page(
        d,
        0,
        [
            _rec(1, long_desc, 3),
            _rec(2, "short", 4),  # dropped: desc < 200
            _rec(3, long_desc, None),  # dropped: null label
            _rec(4, long_desc, 0),  # dropped: invalid label
            _rec(5, long_desc, 3),  # exact dup of rec1 text -> same group
        ],
    )
    return d


class TestLoadPageRecords:
    def test_returns_all_records(self, pages_dir: Path):
        recs = load_page_records(pages_dir)
        assert len(recs) == 5

    def test_reads_multiple_pages(self, pages_dir: Path):
        _write_page(pages_dir, 500, [_rec(6, "B" * 200, 5)])
        recs = load_page_records(pages_dir)
        assert len(recs) == 6


class TestLoadUsableDataset:
    def test_filters_usable_only(self, pages_dir: Path):
        df = load_usable_dataset(pages_dir)
        assert set(df["project_id"]) == {1, 5}

    def test_columns(self, pages_dir: Path):
        df = load_usable_dataset(pages_dir)
        for col in ("project_id", "title", "text", "trl", "group_id", "status", "program"):
            assert col in df.columns

    def test_trl_in_1_9(self, pages_dir: Path):
        df = load_usable_dataset(pages_dir)
        assert df["trl"].between(1, 9).all()

    def test_exact_duplicates_share_group_id(self, pages_dir: Path):
        df = load_usable_dataset(pages_dir)
        g = df.set_index("project_id")["group_id"]
        assert g.loc[1] == g.loc[5]

    def test_unique_descriptions_have_unique_groups(self, pages_dir: Path):
        _write_page(
            pages_dir,
            500,
            [_rec(6, "C" * 200, 5), _rec(7, "D" * 200, 6)],
        )
        df = load_usable_dataset(pages_dir)
        by_id = df.set_index("project_id")["group_id"]
        assert by_id.loc[6] != by_id.loc[7]
        assert by_id.loc[6] != by_id.loc[1]

    def test_text_is_cleaned(self, pages_dir: Path):
        _write_page(pages_dir, 500, [_rec(8, "<p>" + "E" * 200 + "</p>", 2)])
        df = load_usable_dataset(pages_dir)
        row = df[df["project_id"] == 8].iloc[0]
        assert "<p>" not in row["text"]
        assert len(row["text"]) >= 200
