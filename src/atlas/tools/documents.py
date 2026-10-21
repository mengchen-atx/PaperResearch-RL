"""Knowledge-base tools: search_documents and read_document."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from atlas.common.types import Evidence, Modality
from atlas.retrieval.filters import MetadataFilter
from atlas.retrieval.retriever import Retriever
from atlas.tools.base import Tool, ToolArgs, ToolResult


class SearchDocumentsArgs(ToolArgs):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)
    modality: Modality | None = Field(None, description="Restrict to text, table or figure chunks")
    filename: str | None = None


class SearchDocuments(Tool):
    name = "search_documents"
    description = "Hybrid search (BM25 + dense + rerank) over the ingested documents. Returns cited passages."
    Args: ClassVar = SearchDocumentsArgs
    cost = 1.0

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def run(self, args: SearchDocumentsArgs) -> ToolResult:
        flt = MetadataFilter(
            modalities=[args.modality] if args.modality else None,
            filenames=[args.filename] if args.filename else None,
        )
        hits = self.retriever.retrieve(args.query, top_k=args.top_k, flt=flt)
        evidence = [Evidence.from_scored_chunk(h, self.name) for h in hits]
        preview = "\n".join(f"- ({e.citation_label()}) {e.content[:120]}" for e in evidence)
        return ToolResult(tool=self.name, ok=True, output=preview or "no results", evidence=evidence,
                          metadata={"top_score": hits[0].score if hits else None})


class ReadDocumentArgs(ToolArgs):
    filename: str = Field(min_length=1, max_length=300)
    page: int = Field(ge=1)


class ReadDocument(Tool):
    """Read a full page from the index (not the filesystem: no path traversal surface)."""

    name = "read_document"
    description = "Read the full text of one page of an ingested document (by filename and page)."
    Args: ClassVar = ReadDocumentArgs
    cost = 0.5

    def __init__(self, retriever: Retriever) -> None:
        self.chunks = retriever.index.chunks

    def run(self, args: ReadDocumentArgs) -> ToolResult:
        page_chunks = [c for c in self.chunks if c.filename == args.filename and c.page == args.page
                       and c.modality == Modality.TEXT]
        if not page_chunks:
            return ToolResult(tool=self.name, ok=False, error=f"No text for {args.filename} p.{args.page}")
        # Chunks overlap; join them in order and drop exact duplicate sentences.
        seen: set[str] = set()
        parts: list[str] = []
        for c in sorted(page_chunks, key=lambda c: c.chunk_index):
            for sent in c.text.split(". "):
                if sent not in seen:
                    seen.add(sent)
                    parts.append(sent)
        text = ". ".join(parts)
        first = page_chunks[0]
        ev = Evidence(evidence_id=f"{first.doc_id}:page{args.page}", modality=Modality.TEXT, content=text,
                      source_tool=self.name, doc_id=first.doc_id, filename=args.filename, page=args.page)
        return ToolResult(tool=self.name, ok=True, output=text[:500], evidence=[ev])
