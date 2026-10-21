"""Document loaders: file on disk -> (Document, [PageContent]).

PDF text is extracted natively with PyMuPDF (no OCR): born-digital PDFs such as
arXiv papers already contain a text layer, and OCR would be slower and noisier.
Multimodal extraction (tables, figures) lives in `atlas.multimodal.pdf_extract`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from atlas.common.errors import IngestionError
from atlas.common.ids import doc_id_for, sha256_text
from atlas.common.types import Document, PageContent
from atlas.observability.logging import get_logger

log = get_logger("ingestion")

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def file_hash(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pdf(path: Path) -> tuple[Document, list[PageContent]]:
    try:
        pdf = pymupdf.open(path)
    except Exception as exc:  # PyMuPDF raises several error types for corrupt files
        raise IngestionError(f"Could not open PDF '{path.name}': {exc}") from exc
    with pdf:
        if pdf.needs_pass:
            raise IngestionError(f"PDF '{path.name}' is encrypted")
        pages = [
            PageContent(page=i + 1, text=_clean_pdf_text(page.get_text("text")))
            for i, page in enumerate(pdf)
        ]
        title = (pdf.metadata or {}).get("title") or _guess_title(pages)
        doc = Document(
            doc_id=doc_id_for(path.name, file_hash(path)),
            filename=path.name,
            source_path=str(path),
            title=title,
            num_pages=pdf.page_count,
        )
    if not any(p.text.strip() for p in pages):
        # Scanned PDFs have no text layer; OCR would be the right tool there.
        log.warning("PDF has no extractable text (scanned?)", extra={"file": path.name})
    return doc, pages


def load_text(path: Path) -> tuple[Document, list[PageContent]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise IngestionError(f"Could not read '{path.name}': {exc}") from exc
    is_md = path.suffix.lower() in {".md", ".markdown"}
    pages = _split_markdown_sections(text) if is_md else [PageContent(page=None, text=text)]
    doc = Document(
        doc_id=doc_id_for(path.name, sha256_text(text)),
        filename=path.name,
        source_path=str(path),
        title=_guess_title(pages) or path.stem,
        num_pages=None,
    )
    return doc, pages


def load_document(path: Path) -> tuple[Document, list[PageContent]]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestionError(f"Unsupported file type '{suffix}' ({path.name})")
    return load_pdf(path) if suffix == ".pdf" else load_text(path)


def discover_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        raise IngestionError(f"Path does not exist: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


def _split_markdown_sections(text: str) -> list[PageContent]:
    """Markdown has no pages; use headings as the structural unit (section metadata)."""
    sections: list[PageContent] = []
    current_title: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = _MD_HEADING.match(line)
        if m:
            if "".join(buf).strip():
                sections.append(PageContent(page=None, text="\n".join(buf), section=current_title))
            current_title, buf = m.group(2).strip(), [line]
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append(PageContent(page=None, text="\n".join(buf), section=current_title))
    return sections or [PageContent(page=None, text=text)]


def _clean_pdf_text(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line breaks inside words
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _guess_title(pages: list[PageContent]) -> str:
    for page in pages[:1]:
        for line in page.text.splitlines():
            line = line.strip().lstrip("#").strip()
            if len(line) > 8 and not line.lower().startswith(("arxiv", "provided proper")):
                return line[:200]
    return ""
