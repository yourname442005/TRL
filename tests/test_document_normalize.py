"""RED tests for document normalization: headers/footers, page numbers, sections."""

from trl_intelligence.document.loaders import load_document
from trl_intelligence.document.models import Document, Page, Section, TextBlock
from trl_intelligence.document.normalize import normalize


def _multi_page_doc(page_texts: list[str], source_type: str = "pdf") -> Document:
    pages = [
        Page(page_number=i + 1, text=text, blocks=[]) for i, text in enumerate(page_texts)
    ]
    return Document(
        document_id="doc",
        source_path="synthetic",
        source_type=source_type,
        title=None,
        metadata={"loader": "synthetic", "page_count": len(pages)},
        pages=pages,
        sections=[],
        warnings=[],
    )


class TestRepeatedHeaderFooter:
    def test_repeated_header_stripped_from_every_page(self):
        texts = [
            f"ACME CORP PROPRIETARY\n\nbody content unique page {i}" for i in range(1, 6)
        ]
        out = normalize(_multi_page_doc(texts))
        for page in out.pages:
            assert "ACME CORP PROPRIETARY" not in page.text
        assert "body content unique page 3" in out.pages[2].text
        assert out.metadata["removed_repeated_lines"] == 5

    def test_line_on_single_page_is_kept(self):
        texts = [
            "ONLY ON PAGE ONE\n\nbody alpha",
            "body beta text here",
            "body gamma text here",
            "body delta text here",
            "body epsilon text here",
        ]
        out = normalize(_multi_page_doc(texts))
        assert "ONLY ON PAGE ONE" in out.pages[0].text
        assert out.metadata["removed_repeated_lines"] == 0

    def test_metadata_preserves_loader_fields(self):
        texts = [f"ACME\n\nbody {i}" for i in range(1, 6)]
        out = normalize(_multi_page_doc(texts))
        assert out.metadata["loader"] == "synthetic"
        assert out.metadata["page_count"] == 5


class TestPageNumberLines:
    def test_page_number_line_removed(self):
        texts = [
            "body one content",
            "body two content",
            "body three content\n3",
            "body four content",
            "body five content",
        ]
        out = normalize(_multi_page_doc(texts))
        assert out.pages[2].text.splitlines()[-1] != "3"
        assert "3" not in out.pages[2].text.splitlines()
        assert out.metadata["removed_page_number_lines"] == 1

    def test_body_number_not_removed(self):
        texts = [f"body page {i}\nlist item {i}" for i in range(1, 6)]
        out = normalize(_multi_page_doc(texts))
        assert out.metadata["removed_page_number_lines"] == 0
        assert "list item 3" in out.pages[2].text


class TestSectionDetection:
    def test_markdown_heading_paths(self, tmp_path):
        p = tmp_path / "doc.md"
        p.write_text("# Top\n\nIntro paragraph.\n\n## Sub\n\nMore text.", encoding="utf-8")
        doc = normalize(load_document(p))
        paths = [s.path for s in doc.sections]
        assert paths[0] == ["Top"]
        assert paths[1] == ["Top", "Sub"]
        sub = doc.sections[1]
        assert sub.heading == "Sub"
        assert sub.level == 2
        assert sub.page_start == 1 and sub.page_end == 1
        assert len(sub.block_ids) >= 1

    def test_pdf_all_caps_heading_becomes_section(self):
        doc = _multi_page_doc(
            ["TEST RESULTS\n\nThe component passed all checks during evaluation."],
            source_type="pdf",
        )
        out = normalize(doc)
        headings = [s.heading for s in out.sections]
        assert "TEST RESULTS" in headings
        heading_section = next(s for s in out.sections if s.heading == "TEST RESULTS")
        assert heading_section.level == 1
        kinds = [b.kind for b in out.pages[0].blocks]
        assert "heading" in kinds

    def test_plain_text_falls_back_to_body_section(self):
        doc = _multi_page_doc(["just an ordinary sentence without any heading markers."])
        out = normalize(doc)
        assert len(out.sections) == 1
        assert out.sections[0].heading == "body"
        assert out.sections[0].path == ["body"]

    def test_block_offsets_remain_valid_after_normalize(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("first para here\n\nsecond para here", encoding="utf-8")
        doc = normalize(load_document(p))
        block = doc.pages[0].blocks[0]
        assert doc.pages[0].text[block.start_char : block.end_char] == block.text


class TestIdempotence:
    def test_second_normalize_removes_nothing(self):
        texts = [
            f"ACME CORP PROPRIETARY\n\nbody content page {i}\n{i}" for i in range(1, 6)
        ]
        once = normalize(_multi_page_doc(texts))
        twice = normalize(once)
        assert twice.metadata["removed_repeated_lines"] == 0
        assert twice.metadata["removed_page_number_lines"] == 0
        assert [p.text for p in twice.pages] == [p.text for p in once.pages]


def test_sections_and_document_are_serializable():
    import json

    doc = normalize(_multi_page_doc(["body text only"]))
    payload = doc.to_dict()
    assert json.loads(json.dumps(payload))["sections"][0]["heading"] == "body"


def test_section_type_is_importable():
    assert Section is not None and TextBlock is not None
