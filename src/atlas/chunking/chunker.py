"""Sentence-aware chunking with overlap.

Why chunk at all? Embedding models have a limited input window (bge-small: 512
tokens) and one vector per whole paper would blur every topic together. Small
chunks give precise retrieval and precise citations; too-small chunks lose context.

Algorithm (greedy sentence packing):
  1. Split a page into sentences.
  2. Pack consecutive sentences into a chunk until adding one more would exceed
     `chunk_size` characters.
  3. Start the next chunk with the trailing sentences of the previous chunk whose
     total length fits in `chunk_overlap` -> facts that straddle a boundary still
     appear whole in at least one chunk.
  4. Sentences longer than `chunk_size` are hard-split on character windows.

Chunks never cross page boundaries, so every chunk has exactly one page number
for citation. Characters (not tokens) are used as the size unit to keep the
chunker model-agnostic; ~1000 chars is ~250 tokens for English, well under the
embedding model limit.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from atlas.common.config import ChunkingConfig
from atlas.common.ids import chunk_id_for
from atlas.common.types import Chunk, Document, Modality, PageContent

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n{2,}")
_SECTION_NUM = re.compile(r"^\d+(\.\d+){0,3}$")
_SECTION_INLINE = re.compile(r"^(\d+(\.\d+){0,3})\s+([A-Z][A-Za-z\- ,:]{2,80})$")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newlines are PDF line wraps
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


def _hard_split(sentence: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [sentence[i : i + size] for i in range(0, len(sentence), step) if sentence[i : i + size].strip()]


def pack_sentences(sentences: list[str], size: int, overlap: int) -> list[str]:
    """Greedy packing of sentences into overlapping windows (pure function)."""
    units: list[str] = []
    for s in sentences:
        units.extend(_hard_split(s, size, overlap) if len(s) > size else [s])

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        added = len(unit) + (1 if current else 0)
        if current and current_len + added > size:
            chunks.append(" ".join(current))
            # carry over the tail of the previous chunk as overlap
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) + 1 > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current, current_len = tail, max(0, tail_len - 1)
            added = len(unit) + (1 if current else 0)
        current.append(unit)
        current_len += added
    if current:
        chunks.append(" ".join(current))
    return chunks


def detect_sections(page_text: str) -> list[tuple[int, str]]:
    """Find numbered section headings -> [(char_offset, 'N.M Title')]."""
    found: list[tuple[int, str]] = []
    lines = page_text.split("\n")
    offset = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        m = _SECTION_INLINE.match(stripped)
        if m:
            found.append((offset, f"{m.group(1)} {m.group(3).strip()}"))
        elif _SECTION_NUM.match(stripped) and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if 2 < len(nxt) <= 80 and nxt[:1].isupper() and not nxt.endswith("."):
                found.append((offset, f"{stripped} {nxt}"))
        offset += len(line) + 1
    return found


class Chunker:
    def __init__(self, cfg: ChunkingConfig) -> None:
        self.cfg = cfg

    def chunk_document(self, doc: Document, pages: Iterable[PageContent]) -> list[Chunk]:
        chunks: list[Chunk] = []
        section: str | None = None
        for page in pages:
            section = page.section or section
            sections = detect_sections(page.text)
            # Chunk each section span separately so section metadata is accurate.
            boundaries = [(0, section)] + [(off, title) for off, title in sections if off > 0]
            if sections and sections[0][0] == 0:
                boundaries[0] = (0, sections[0][1])
            for idx, (start, title) in enumerate(boundaries):
                end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(page.text)
                section = title or section
                span = page.text[start:end]
                for text in pack_sentences(split_sentences(span), self.cfg.chunk_size, self.cfg.chunk_overlap):
                    if len(text) < self.cfg.min_chunk_chars:
                        continue
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id_for(doc.doc_id, "t", len(chunks), text),
                            doc_id=doc.doc_id,
                            filename=doc.filename,
                            text=text,
                            modality=Modality.TEXT,
                            page=page.page,
                            section=section,
                            chunk_index=len(chunks),
                            metadata={"title": doc.title},
                        )
                    )
        return chunks
