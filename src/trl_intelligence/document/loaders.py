"""Source loaders: PDF (page-preserving), plain text, Markdown."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from trl_intelligence.document.models import (
    Document,
    DocumentLoadError,
    Page,
    TextBlock,
    UnsupportedFormatError,
)
from trl_intelligence.preprocessing.text import clean_text

_LOADER_VERSION = "1.0.0"
_TXT_SUFFIXES = {".txt", ".text"}
_MD_SUFFIXES = {".md", ".markdown"}
_HEADING_RE = re.compile(r"^#{1,6}\s+\S")


def load_document(path: str | Path, document_id: str | None = None) -> Document:
    """Load a source file into a normalized Document (no normalization applied yet)."""
    src = Path(path)
    if not src.exists():
        raise DocumentLoadError(f"file not found: {src}", source_path=str(src))
    suffix = src.suffix.lower()
    if suffix == ".pdf":
        loader = _load_pdf
    elif suffix in _TXT_SUFFIXES:
        loader = _load_text
    elif suffix in _MD_SUFFIXES:
        loader = _load_markdown
    else:
        raise UnsupportedFormatError(
            f"unsupported format: {suffix or '(no suffix)'}", source_path=str(src)
        )
    raw = src.read_bytes()
    doc_id = document_id if document_id is not None else hashlib.sha1(raw).hexdigest()[:16]
    try:
        return loader(src, raw, doc_id)
    except DocumentLoadError:
        raise
    except Exception as exc:  # malformed source -> controlled error
        raise DocumentLoadError(f"cannot parse {suffix}: {exc}", source_path=str(src)) from exc


def _base_metadata(loader: str, src: Path, raw: bytes, page_count: int) -> dict:
    return {
        "loader": loader,
        "loader_version": _LOADER_VERSION,
        "byte_size": len(raw),
        "page_count": page_count,
    }


def document_from_text(
    text: str, document_id: str | None = None, source_path: str = "<text>"
) -> Document:
    """Build a one-page txt Document from an in-memory string (no file needed)."""
    doc_id = (
        document_id
        if document_id is not None
        else hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    )
    warnings = ["empty_document"] if not text.strip() else []
    blocks = _paragraph_blocks(text, page_number=1)
    page = Page(page_number=1, text=text, blocks=blocks)
    metadata = {
        "loader": "text",
        "loader_version": _LOADER_VERSION,
        "byte_size": len(text.encode("utf-8")),
        "page_count": 1,
    }
    return Document(
        document_id=doc_id,
        source_path=source_path,
        source_type="txt",
        title=None,
        metadata=metadata,
        pages=[page],
        sections=[],
        warnings=warnings,
    )


def _load_text(src: Path, raw: bytes, doc_id: str) -> Document:
    text = raw.decode("utf-8", errors="replace")
    warnings = ["empty_document"] if not text.strip() else []
    blocks = _paragraph_blocks(text, page_number=1)
    page = Page(page_number=1, text=text, blocks=blocks)
    metadata = _base_metadata("text", src, raw, 1)
    return Document(
        document_id=doc_id,
        source_path=str(src),
        source_type="txt",
        title=src.stem,
        metadata=metadata,
        pages=[page],
        sections=[],
        warnings=warnings,
    )


def _paragraph_blocks(text: str, page_number: int) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    idx = 0
    for match in re.finditer(r"(.*?)(\n{2,}|$)", text, re.DOTALL):
        para = match.group(1)
        if not para.strip():
            continue
        leading = len(para) - len(para.lstrip())
        start = match.start(1) + leading
        stripped = para.strip()
        end = start + len(stripped)
        blocks.append(
            TextBlock(
                block_id=f"p{page_number}:b{idx:02d}",
                kind="paragraph",
                text=stripped,
                start_char=start,
                end_char=end,
            )
        )
        idx += 1
    return blocks


def _load_markdown(src: Path, raw: bytes, doc_id: str) -> Document:
    text = raw.decode("utf-8", errors="replace")
    warnings = ["empty_document"] if not text.strip() else []
    blocks = _markdown_blocks(text, page_number=1)
    page = Page(page_number=1, text=text, blocks=blocks)
    metadata = _base_metadata("markdown", src, raw, 1)
    return Document(
        document_id=doc_id,
        source_path=str(src),
        source_type="md",
        title=src.stem,
        metadata=metadata,
        pages=[page],
        sections=[],
        warnings=warnings,
    )


def _markdown_blocks(text: str, page_number: int) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    paragraph_lines: list[str] = []
    paragraph_start = 0

    def flush_paragraph(end_line_start: int) -> None:
        if not paragraph_lines:
            return
        joined = " ".join(line.strip() for line in paragraph_lines).strip()
        if joined:
            start = paragraph_start
            blocks.append(
                TextBlock(
                    block_id=f"p{page_number}:b{len(blocks):02d}",
                    kind="paragraph",
                    text=joined,
                    start_char=start,
                    end_char=start + len(joined),
                )
            )
        paragraph_lines.clear()

    offset = 0
    for line in text.splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        stripped = line.strip()
        if not stripped:
            flush_paragraph(line_start)
            continue
        if _HEADING_RE.match(stripped):
            flush_paragraph(line_start)
            blocks.append(
                TextBlock(
                    block_id=f"p{page_number}:b{len(blocks):02d}",
                    kind="heading",
                    text=stripped,
                    start_char=line_start,
                    end_char=line_start + len(stripped),
                )
            )
            continue
        if not paragraph_lines:
            paragraph_start = line_start
        paragraph_lines.append(stripped)
    flush_paragraph(offset)
    return blocks


def _load_pdf(src: Path, raw: bytes, doc_id: str) -> Document:
    import pymupdf

    try:
        pdf = pymupdf.open(str(src))
    except Exception as exc:
        raise DocumentLoadError(f"cannot parse pdf: {exc}", source_path=str(src)) from exc

    with pdf:
        pages: list[Page] = []
        warnings: list[str] = []
        for index in range(pdf.page_count):
            page_number = index + 1
            raw_blocks = pdf[index].get_text("blocks")
            texts = [
                clean_text(item[4])
                for item in sorted(raw_blocks, key=lambda b: (b[1], b[0]))
                if item[6] == 0 and clean_text(item[4])
            ]
            joined = "\n\n".join(texts)
            blocks: list[TextBlock] = []
            cursor = 0
            for block_index, block_text in enumerate(texts):
                start = joined.find(block_text, cursor)
                if start < 0:
                    start = cursor
                end = start + len(block_text)
                cursor = end
                blocks.append(
                    TextBlock(
                        block_id=f"p{page_number}:b{block_index:02d}",
                        kind="paragraph",
                        text=block_text,
                        start_char=start,
                        end_char=end,
                    )
                )
            if not joined.strip():
                warnings.append(f"page_{page_number}_no_text")
            pages.append(Page(page_number=page_number, text=joined, blocks=blocks))

        if pages and all(not page.text.strip() for page in pages):
            warnings = ["no_extractable_text", "possibly_scanned_image_only"]
        if not pages:
            warnings = ["empty_document"]

        metadata = _base_metadata("pymupdf", src, raw, len(pages))
        return Document(
            document_id=doc_id,
            source_path=str(src),
            source_type="pdf",
            title=src.stem,
            metadata=metadata,
            pages=pages,
            sections=[],
            warnings=warnings,
        )
