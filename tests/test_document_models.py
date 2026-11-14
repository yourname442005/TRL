"""RED tests for the document schema (Milestone 2 Task 1)."""

import pytest

from trl_intelligence.document.models import (
    Document,
    DocumentLoadError,
    Page,
    Section,
    TextBlock,
)


class TestTextBlock:
    def test_fields_and_offsets(self):
        b = TextBlock(block_id="p1:b00", kind="paragraph", text="hello world", start_char=0, end_char=11)
        assert (b.start_char, b.end_char) == (0, 11)
        assert b.kind == "paragraph"

    def test_kind_must_be_known(self):
        with pytest.raises(ValueError):
            TextBlock(block_id="p1:b00", kind="footnote", text="x", start_char=0, end_char=1)


class TestPage:
    def test_page_numbers_are_one_based(self):
        with pytest.raises(ValueError):
            Page(page_number=0, text="x", blocks=[])

    def test_valid_page(self):
        p = Page(page_number=1, text="abc", blocks=[])
        assert p.page_number == 1


class TestSection:
    def test_section_fields(self):
        s = Section(
            section_id="s0",
            heading="body",
            level=0,
            path=["body"],
            page_start=1,
            page_end=1,
            block_ids=[],
        )
        assert s.path == ["body"]


class TestDocument:
    def _doc(self) -> Document:
        return Document(
            document_id="d1",
            source_path="x.txt",
            source_type="txt",
            title=None,
            metadata={},
            pages=[Page(page_number=1, text="abc", blocks=[])],
            sections=[
                Section(
                    section_id="s0",
                    heading="body",
                    level=0,
                    path=["body"],
                    page_start=1,
                    page_end=1,
                    block_ids=[],
                )
            ],
            warnings=[],
        )

    def test_page_count_and_text(self):
        doc = self._doc()
        assert doc.page_count == 1
        assert doc.text == "abc"

    def test_text_joins_pages(self):
        doc = self._doc()
        doc = Document(
            document_id=doc.document_id,
            source_path=doc.source_path,
            source_type=doc.source_type,
            title=doc.title,
            metadata=doc.metadata,
            pages=[
                Page(page_number=1, text="one", blocks=[]),
                Page(page_number=2, text="two", blocks=[]),
            ],
            sections=doc.sections,
            warnings=doc.warnings,
        )
        assert doc.page_count == 2
        assert doc.text == "one\ntwo"

    def test_source_type_validated(self):
        with pytest.raises(ValueError):
            Document(
                document_id="d",
                source_path="x.docx",
                source_type="docx",
                title=None,
                metadata={},
                pages=[],
                sections=[],
                warnings=[],
            )

    def test_to_dict_is_json_safe(self):
        import json

        payload = self._doc().to_dict()
        assert json.loads(json.dumps(payload))["document_id"] == "d1"
        assert payload["page_count"] == 1


class TestDocumentLoadError:
    def test_carries_reason_and_path(self):
        err = DocumentLoadError("bad pdf", source_path="a.pdf")
        assert "bad pdf" in str(err)
        assert err.source_path == "a.pdf"
