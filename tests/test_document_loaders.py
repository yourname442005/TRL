"""RED tests for document loaders: txt, md, pdf (Milestone 2 Task 2)."""

import json
from pathlib import Path

import pytest

from trl_intelligence.document.loaders import load_document
from trl_intelligence.document.models import DocumentLoadError, UnsupportedFormatError


def _make_pdf(path: Path, pages: list[list[str]]) -> None:
    import pymupdf

    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line)
            y += 20
    doc.save(str(path))
    doc.close()


class TestDispatch:
    def test_txt_suffix(self, tmp_path: Path):
        p = tmp_path / "a.txt"
        p.write_text("hello world", encoding="utf-8")
        assert load_document(p).source_type == "txt"

    def test_md_suffixes(self, tmp_path: Path):
        for name in ("a.md", "a.markdown"):
            p = tmp_path / name
            p.write_text("# Title", encoding="utf-8")
            assert load_document(p).source_type == "md"

    def test_pdf_suffix(self, tmp_path: Path):
        p = tmp_path / "a.pdf"
        _make_pdf(p, [["page one text"]])
        assert load_document(p).source_type == "pdf"

    def test_unsupported_suffix_raises(self, tmp_path: Path):
        p = tmp_path / "a.csv"
        p.write_text("a,b", encoding="utf-8")
        with pytest.raises(UnsupportedFormatError):
            load_document(p)


class TestTextLoader:
    def test_empty_document(self, tmp_path: Path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        doc = load_document(p)
        assert doc.warnings == ["empty_document"]
        assert doc.page_count == 1
        assert doc.text == ""
        assert doc.blocks_of_page(1) == []

    def test_paragraph_blocks_with_offsets(self, tmp_path: Path):
        p = tmp_path / "multi.txt"
        p.write_text("first paragraph here\n\nsecond paragraph here", encoding="utf-8")
        doc = load_document(p)
        blocks = doc.blocks_of_page(1)
        assert [b.block_id for b in blocks] == ["p1:b00", "p1:b01"]
        assert blocks[0].text == "first paragraph here"
        assert blocks[1].text == "second paragraph here"
        # offsets ascend and index into page text
        assert blocks[0].start_char == 0
        assert blocks[1].start_char > blocks[0].end_char
        page = doc.pages[0]
        assert page.text[blocks[0].start_char : blocks[0].end_char] == blocks[0].text

    def test_document_id_is_content_hash(self, tmp_path: Path):
        p = tmp_path / "same.txt"
        p.write_text("stable content", encoding="utf-8")
        doc1 = load_document(p)
        doc2 = load_document(p)
        assert doc1.document_id == doc2.document_id
        assert len(doc1.document_id) == 16


class TestMarkdownLoader:
    def test_heading_blocks(self, tmp_path: Path):
        p = tmp_path / "doc.md"
        p.write_text("# Top\n\nSome intro text.\n\n## Sub\n\nMore text.", encoding="utf-8")
        doc = load_document(p)
        blocks = doc.blocks_of_page(1)
        kinds = [b.kind for b in blocks]
        assert kinds[0] == "heading"
        assert blocks[0].text.startswith("# Top")
        assert "heading" in kinds  # ## Sub also marked as heading
        sub = [b for b in blocks if b.text.startswith("## Sub")]
        assert len(sub) == 1
        assert sub[0].kind == "heading"


class TestPdfLoader:
    def test_page_boundaries_preserved(self, tmp_path: Path):
        p = tmp_path / "two.pdf"
        _make_pdf(p, [["alpha unique text"], ["beta unique text"]])
        doc = load_document(p)
        assert doc.page_count == 2
        assert "alpha" in doc.pages[0].text
        assert "beta" not in doc.pages[0].text
        assert "beta" in doc.pages[1].text
        assert "alpha" not in doc.pages[1].text

    def test_blocks_have_page_scoped_ids(self, tmp_path: Path):
        p = tmp_path / "ids.pdf"
        _make_pdf(p, [["line one"], ["line two"]])
        doc = load_document(p)
        assert doc.pages[0].blocks[0].block_id.startswith("p1:")
        assert doc.pages[1].blocks[0].block_id.startswith("p2:")

    def test_image_only_pdf_flagged(self, tmp_path: Path):
        import pymupdf

        p = tmp_path / "image.pdf"
        doc_raw = pymupdf.open()
        page = doc_raw.new_page()
        page.draw_rect(pymupdf.Rect(50, 50, 200, 200))
        doc_raw.save(str(p))
        doc_raw.close()
        doc = load_document(p)
        assert "no_extractable_text" in doc.warnings
        assert "possibly_scanned_image_only" in doc.warnings
        assert doc.pages[0].text == ""

    def test_malformed_pdf_raises_load_error(self, tmp_path: Path):
        p = tmp_path / "bad.pdf"
        p.write_bytes(b"not a pdf at all")
        with pytest.raises(DocumentLoadError) as exc:
            load_document(p)
        assert exc.value.source_path == str(p)

    def test_pdf_metadata(self, tmp_path: Path):
        p = tmp_path / "meta.pdf"
        _make_pdf(p, [["x"]])
        doc = load_document(p)
        assert doc.metadata["loader"] == "pymupdf"
        assert doc.metadata["page_count"] == 1
        assert doc.title == "meta"


class TestMetadata:
    def test_txt_metadata(self, tmp_path: Path):
        p = tmp_path / "m.txt"
        p.write_text("content", encoding="utf-8")
        doc = load_document(p)
        assert doc.metadata["loader"] == "text"
        assert doc.metadata["byte_size"] == len("content")
        assert doc.title == "m"


def test_json_round_trip_of_metadata(tmp_path: Path):
    p = tmp_path / "j.txt"
    p.write_text("x", encoding="utf-8")
    doc = load_document(p)
    assert json.loads(json.dumps(doc.to_dict()))["source_type"] == "txt"
