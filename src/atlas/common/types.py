"""Core domain types shared by every layer of Atlas.

The pipeline is a chain of transformations over these objects:

    Document --(chunking)--> Chunk --(retrieval)--> ScoredChunk --(agent/tools)--> Evidence

All models are frozen (immutable). Stages return new objects instead of mutating
inputs, which keeps the retrieval/agent pipeline free of hidden side effects and
makes trajectories safe to serialize at any point.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Modality(StrEnum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    WEB = "web"
    COMPUTATION = "computation"


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Document(Frozen):
    """A parsed source document (PDF, text, markdown)."""

    doc_id: str
    filename: str
    source_path: str
    title: str = ""
    num_pages: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageContent(Frozen):
    """Raw content extracted from one page (or the whole file for text formats)."""

    page: int | None  # 1-based page number; None for formats without pages
    text: str
    section: str | None = None


class Chunk(Frozen):
    """The unit that gets embedded, indexed and retrieved.

    A chunk carries enough metadata to produce a verifiable citation:
    document, page, section, and (for multimodal chunks) table/figure identity.
    """

    chunk_id: str
    doc_id: str
    filename: str
    text: str
    modality: Modality = Modality.TEXT
    page: int | None = None
    section: str | None = None
    chunk_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def citation_label(self) -> str:
        parts = [self.filename]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.modality == Modality.TABLE and "table_id" in self.metadata:
            parts.append(str(self.metadata["table_id"]))
        if self.modality == Modality.FIGURE and "figure_id" in self.metadata:
            parts.append(str(self.metadata["figure_id"]))
        return ", ".join(parts)


class ScoredChunk(Frozen):
    """A retrieved chunk plus the score that ranked it.

    `component_scores` keeps per-retriever scores (e.g. bm25, dense, rerank) so
    ablations can inspect *why* a chunk ranked where it did.
    """

    chunk: Chunk
    score: float
    retriever: str
    component_scores: dict[str, float] = Field(default_factory=dict)


class Evidence(Frozen):
    """A unit of support collected by the agent, from any tool or modality."""

    evidence_id: str
    modality: Modality
    content: str
    source_tool: str
    doc_id: str | None = None
    filename: str | None = None
    page: int | None = None
    score: float | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def citation_label(self) -> str:
        if self.url:
            return self.url
        parts = [self.filename or self.source_tool]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        for key in ("table_id", "figure_id"):
            if key in self.metadata:
                parts.append(str(self.metadata[key]))
        return ", ".join(parts)

    @classmethod
    def from_scored_chunk(cls, sc: ScoredChunk, source_tool: str) -> Evidence:
        c = sc.chunk
        return cls(
            evidence_id=c.chunk_id,
            modality=c.modality,
            content=c.text,
            source_tool=source_tool,
            doc_id=c.doc_id,
            filename=c.filename,
            page=c.page,
            score=sc.score,
            metadata={k: v for k, v in c.metadata.items() if isinstance(v, str | int | float | bool)},
        )


class Citation(Frozen):
    """A resolved citation marker `[n]` in a generated answer."""

    marker: int
    evidence_id: str
    label: str
    quote: str = ""


class TokenUsage(Frozen):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )
