"""Ingestion pipeline: files -> documents -> chunks (text/table/figure) -> index.

A malformed file must not abort ingestion of an entire corpus: failures are
collected per file and reported in the returned `IngestReport`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from atlas.chunking.chunker import Chunker
from atlas.common.config import AtlasConfig
from atlas.common.errors import IngestionError
from atlas.common.types import Chunk, Document
from atlas.embeddings.embedder import Embedder
from atlas.ingestion.loaders import discover_files, load_document
from atlas.multimodal.pdf_extract import extract_multimodal, multimodal_chunks
from atlas.multimodal.table_store import TableStore
from atlas.observability.logging import get_logger
from atlas.retrieval.index import KnowledgeIndex, build_index, save_index

log = get_logger("ingest")


@dataclass
class IngestReport:
    documents: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    figure_chunks: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    def summary(self) -> str:
        s = (f"{self.documents} documents -> {self.text_chunks} text / {self.table_chunks} table / "
             f"{self.figure_chunks} figure chunks in {self.seconds:.1f}s")
        if self.failed:
            s += f"; {len(self.failed)} failed: " + ", ".join(self.failed)
        return s


def ingest_paths(
    cfg: AtlasConfig, embedder: Embedder, sources: list[Path], index_dir: Path | None = None
) -> tuple[KnowledgeIndex, IngestReport]:
    t0 = time.perf_counter()
    index_dir = index_dir or cfg.resolve_path(cfg.paths.index_dir)
    report = IngestReport()
    chunker = Chunker(cfg.chunking)
    table_store = TableStore(index_dir / "tables.sqlite")
    table_store.reset()
    figure_dir = index_dir / "figures"

    files: list[Path] = []
    for src in sources:
        files.extend(discover_files(src))
    if not files:
        raise IngestionError(f"No supported documents (.pdf/.txt/.md) found in {', '.join(map(str, sources))}")

    documents: list[Document] = []
    chunks: list[Chunk] = []
    for path in files:
        try:
            doc, pages = load_document(path)
        except IngestionError as exc:
            log.warning("skipping file", extra={"file": path.name, "error": str(exc)})
            report.failed[path.name] = str(exc)
            continue
        doc_chunks = chunker.chunk_document(doc, pages)
        if path.suffix.lower() == ".pdf":
            mm = extract_multimodal(path, doc, cfg.multimodal, figure_dir)
            mm_chunks = multimodal_chunks(doc, mm, start_index=len(doc_chunks))
            table_store.add_tables(doc, mm.tables)
            report.table_chunks += len(mm.tables)
            report.figure_chunks += len(mm.figures)
            doc_chunks = doc_chunks + mm_chunks
        report.text_chunks += sum(1 for c in doc_chunks if c.modality == "text")
        documents.append(doc)
        chunks.extend(doc_chunks)
        log.info("parsed", extra={"file": path.name, "chunks": len(doc_chunks)})

    if not chunks:
        raise IngestionError("Ingestion produced no chunks; all documents were empty or failed.")
    index = build_index(cfg, embedder, documents, chunks)
    save_index(index, index_dir)
    report.documents = len(documents)
    report.seconds = time.perf_counter() - t0
    return index, report
