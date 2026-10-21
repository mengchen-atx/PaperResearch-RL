"""Multimodal PDF extraction: tables and figures, without OCR.

Routing decision per content type:
  * Running text  -> native text layer (atlas.ingestion.loaders)
  * Tables        -> PyMuPDF `find_tables()` on ruled tables -> markdown chunk for
                     retrieval + structured rows in SQLite for the SQL tool
  * Figures/charts-> located via their "Figure N:" caption; the region above the
                     caption is rendered to PNG. The *caption* (native text) is
                     indexed immediately; a VLM is only invoked when a question
                     needs visual reasoning (see `atlas.tools.visual`) or when
                     `describe_figures_at_ingest` is enabled.

This keeps ingestion fast and cheap, and reserves the expensive VLM for content
that genuinely requires looking at pixels (e.g. reading a curve off a chart).

Table detection on real papers is imperfect (borderless tables are missed, some
dense figures are mis-detected as tables), so detected tables are filtered by
shape/density heuristics below. Missed tables still exist in the text chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from atlas.common.config import MultimodalConfig
from atlas.common.ids import chunk_id_for
from atlas.common.types import Chunk, Document, Modality
from atlas.observability.logging import get_logger

log = get_logger("multimodal.extract")

# Captions: "Figure 3: ..." / "Figure 3. ..." / "Figure 3" on its own line (some templates).
_FIG_CAPTION = re.compile(r"^\s*(Figure|Fig\.)\s*(\d+)\s*(?:[:.]|\n|$)", re.IGNORECASE)
_TABLE_CAPTION = re.compile(r"^\s*Table\s*(\d+)\s*(?:[:.]|\n|$)", re.IGNORECASE)

MAX_TABLE_COLS = 16
MIN_FILL_RATIO = 0.5
MAX_MEAN_CELL_CHARS = 120
MIN_UNCAPTIONED_CELLS = 6
MAX_TINY_CELL_RATIO = 0.5  # diagrams mis-detected as tables are mostly 1-2 char cells
FIGURE_RENDER_DPI = 110


@dataclass(frozen=True)
class ExtractedTable:
    table_id: str
    page: int
    caption: str
    header: list[str]
    rows: list[list[str]]
    markdown: str


@dataclass(frozen=True)
class ExtractedFigure:
    figure_id: str
    page: int
    caption: str
    image_path: str
    bbox: tuple[float, float, float, float]


@dataclass
class MultimodalExtraction:
    tables: list[ExtractedTable] = field(default_factory=list)
    figures: list[ExtractedFigure] = field(default_factory=list)


def _clean_cell(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _expand_multiline_rows(raw_rows: list[list[str]]) -> list[list[str]]:
    """PyMuPDF often merges visually separate rows into one cell with newlines.

    If every cell in a row splits into the same number of lines, un-merge them.
    Otherwise keep the row as-is (collapsed whitespace) rather than guess.
    """
    out: list[list[str]] = []
    for row in raw_rows:
        parts = [str(c or "").split("\n") for c in row]
        counts = {len(p) for p in parts if any(x.strip() for x in p)}
        if len(counts) == 1 and next(iter(counts)) > 1:
            n = next(iter(counts))
            for i in range(n):
                out.append([_clean_cell(p[i]) if len(p) == n else "" for p in parts])
        else:
            out.append([_clean_cell(c) for c in row])
    return out


def _is_plausible_table(rows: list[list[str]], has_caption: bool = True) -> bool:
    if len(rows) < 2 or not rows[0] or len(rows[0]) < 2 or len(rows[0]) > MAX_TABLE_COLS:
        return False
    cells = [c for r in rows for c in r]
    filled = [c for c in cells if c]
    if not filled or len(filled) / len(cells) < MIN_FILL_RATIO:
        return False
    if sum(len(c) for c in filled) / len(filled) > MAX_MEAN_CELL_CHARS:
        return False
    if not has_caption:
        # Without a "Table N" caption, demand more evidence that this is a real table.
        tiny = sum(1 for c in filled if len(c) <= 2) / len(filled)
        if len(filled) < MIN_UNCAPTIONED_CELLS or tiny > MAX_TINY_CELL_RATIO:
            return False
    return True


def _to_markdown(header: list[str], rows: list[list[str]]) -> str:
    def fmt(r: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "/") for c in r) + " |"

    lines = [fmt(header), "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend(fmt(r + [""] * (len(header) - len(r))) for r in rows)
    return "\n".join(lines)


def _nearest_caption(page: pymupdf.Page, bbox: pymupdf.Rect, pattern: re.Pattern) -> str:
    """Find the caption block closest (vertically) to a table/figure region."""
    best, best_dist = "", float("inf")
    for block in page.get_text("blocks"):
        y0, y1, text = block[1], block[3], block[4]
        if not pattern.match(text):
            continue
        dist = min(abs(y0 - bbox.y1), abs(bbox.y0 - y1))
        if dist < best_dist:
            best, best_dist = re.sub(r"\s+", " ", text).strip(), dist
    return best if best_dist < 120 else ""


def extract_tables(page: pymupdf.Page, doc: Document) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    try:
        found = page.find_tables()
    except Exception:  # table finder can fail on malformed pages; text is still indexed
        log.warning("table detection failed", extra={"file": doc.filename, "page": page.number + 1})
        return tables
    for t in found.tables:
        rows = _expand_multiline_rows(t.extract())
        caption = _nearest_caption(page, pymupdf.Rect(t.bbox), _TABLE_CAPTION)
        if not _is_plausible_table(rows, has_caption=bool(caption)):
            continue
        header = [h or f"col{i + 1}" for i, h in enumerate(rows[0])]
        body = rows[1:]
        m = re.match(r"\s*Table\s*(\d+)", caption)
        table_id = f"Table {m.group(1)}" if m else f"Table p{page.number + 1}-{len(tables) + 1}"
        if any(t_.table_id == table_id for t_ in tables):  # one caption, several detected parts
            table_id = f"{table_id} (part {sum(t_.table_id.startswith(table_id) for t_ in tables) + 1})"
        tables.append(
            ExtractedTable(
                table_id=table_id,
                page=page.number + 1,
                caption=caption,
                header=header,
                rows=body,
                markdown=_to_markdown(header, body),
            )
        )
    return tables


def extract_figures(page: pymupdf.Page, doc: Document, out_dir: Path) -> list[ExtractedFigure]:
    """Locate figures by caption and render the graphical region above each caption."""
    figures: list[ExtractedFigure] = []
    blocks = page.get_text("blocks")
    graphics = [pymupdf.Rect(d["rect"]) for d in page.get_drawings()]
    graphics += [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
    graphics = [g for g in graphics if g.width > 2 and g.height > 2]
    text_bottoms = sorted(b[3] for b in blocks)

    for bi, block in enumerate(blocks):
        text = block[4]
        m = _FIG_CAPTION.match(text)
        if not m:
            continue
        if len(text.strip()) < 20 and bi + 1 < len(blocks):  # caption body in the next block
            text = f"{text.strip()}: {blocks[bi + 1][4]}"
        cap_rect = pymupdf.Rect(block[:4])
        above = [g for g in graphics if g.y1 <= cap_rect.y0 + 2 and g.y0 >= page.rect.y0]
        # Only keep graphics below the last text block that ends well above them,
        # i.e. the contiguous graphical region directly on top of the caption.
        if above:
            region = pymupdf.Rect(above[0])
            for g in above[1:]:
                region |= g
            prior_text = [y for y in text_bottoms if y < region.y0 - 1]
            region.y0 = max(region.y0, prior_text[-1] if prior_text else page.rect.y0)
        else:
            region = pymupdf.Rect(page.rect.x0, page.rect.y0, page.rect.x1, cap_rect.y0)
        region = (region | pymupdf.Rect(cap_rect.x0, region.y0, cap_rect.x1, cap_rect.y0)) & page.rect
        if region.width * region.height < 2000:
            continue
        figure_id = f"Figure {m.group(2)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir / f"{doc.doc_id}_p{page.number + 1}_fig{m.group(2)}.png"
        page.get_pixmap(clip=region, dpi=FIGURE_RENDER_DPI).save(img_path)
        figures.append(
            ExtractedFigure(
                figure_id=figure_id,
                page=page.number + 1,
                caption=re.sub(r"\s+", " ", text).strip(),
                image_path=str(img_path),
                bbox=(region.x0, region.y0, region.x1, region.y1),
            )
        )
    return figures


def extract_multimodal(pdf_path: Path, doc: Document, cfg: MultimodalConfig, figure_dir: Path) -> MultimodalExtraction:
    result = MultimodalExtraction()
    try:
        pdf = pymupdf.open(pdf_path)
    except Exception:
        log.warning("cannot open PDF for multimodal extraction", extra={"file": doc.filename})
        return result
    with pdf:
        for page in pdf:
            if cfg.extract_tables:
                result.tables.extend(extract_tables(page, doc))
            if cfg.extract_figures:
                result.figures.extend(extract_figures(page, doc, figure_dir))
    return result


def multimodal_chunks(doc: Document, extraction: MultimodalExtraction, start_index: int) -> list[Chunk]:
    """Convert extracted tables/figures into retrievable chunks with modality metadata."""
    chunks: list[Chunk] = []
    idx = start_index
    for t in extraction.tables:
        text = f"{t.caption}\n{t.markdown}".strip() if t.caption else f"{t.table_id}\n{t.markdown}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id_for(doc.doc_id, "tab", idx, text),
                doc_id=doc.doc_id,
                filename=doc.filename,
                text=text,
                modality=Modality.TABLE,
                page=t.page,
                chunk_index=idx,
                metadata={"table_id": t.table_id, "caption": t.caption, "title": doc.title},
            )
        )
        idx += 1
    for f in extraction.figures:
        text = f"{f.figure_id} (image). {f.caption}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id_for(doc.doc_id, "fig", idx, text),
                doc_id=doc.doc_id,
                filename=doc.filename,
                text=text,
                modality=Modality.FIGURE,
                page=f.page,
                chunk_index=idx,
                metadata={"figure_id": f.figure_id, "caption": f.caption, "image_path": f.image_path,
                          "title": doc.title},
            )
        )
        idx += 1
    return chunks
