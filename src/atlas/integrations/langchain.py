"""LangChain adapters for Atlas components.

Atlas's retrieval stack and tools are implemented from scratch on purpose; this
module does NOT replace them. It is a thin interoperability layer that exposes
them through LangChain's standard interfaces, so they can be dropped into any
LangChain / LangGraph application (chains, `create_react_agent`, tool-calling
chat models) without that application knowing about Atlas internals.

    AtlasRetriever      -> langchain_core.retrievers.BaseRetriever
                           (query -> list[Document] with citation metadata)
    to_langchain_tool   -> langchain_core.tools.StructuredTool
                           (same pydantic argument schema, executed through
                            Atlas's ToolExecutor so timeouts, metrics and
                            tracing still apply)

Example:
    rt = AtlasRuntime(load_config())
    docs = AtlasRetriever(atlas_retriever=rt.retriever).invoke("What is BM25?")
    tools = langchain_tools(rt.executor)          # bind to any tool-calling LLM
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import StructuredTool, ToolException
from pydantic import ConfigDict

from atlas.common.types import Evidence, ScoredChunk
from atlas.retrieval.filters import MetadataFilter
from atlas.retrieval.retriever import Retriever
from atlas.tools.base import Tool, ToolExecutor


def scored_chunk_to_document(sc: ScoredChunk) -> Document:
    c = sc.chunk
    return Document(
        id=c.chunk_id,
        page_content=c.text,
        metadata={
            "chunk_id": c.chunk_id, "doc_id": c.doc_id, "filename": c.filename, "page": c.page,
            "section": c.section, "modality": str(c.modality), "score": sc.score, "retriever": sc.retriever,
            "citation": c.citation_label(),
        },
    )


class AtlasRetriever(BaseRetriever):
    """Atlas hybrid retrieval (BM25 + dense + RRF + optional rerank) as a LangChain retriever."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    atlas_retriever: Retriever
    top_k: int = 5
    metadata_filter: MetadataFilter | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        hits = self.atlas_retriever.retrieve(query, top_k=self.top_k, flt=self.metadata_filter)
        return [scored_chunk_to_document(h) for h in hits]


def _format_evidence(evidence: list[Evidence]) -> str:
    return "\n".join(f"[{i}] ({e.citation_label()}) {e.content[:500]}" for i, e in enumerate(evidence, start=1))


def to_langchain_tool(tool: Tool, executor: ToolExecutor) -> StructuredTool:
    """Wrap one Atlas tool. Content = readable evidence for the LLM; artifact = structured evidence."""

    def run(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        result, _record = executor.call(tool.name, kwargs)
        if not result.ok:
            # Surfaced to the calling agent as a tool error message instead of crashing it.
            raise ToolException(result.error or f"{tool.name} failed")
        content = _format_evidence(result.evidence) if result.evidence else (result.output or "no results")
        return content, [e.model_dump(mode="json") for e in result.evidence]

    return StructuredTool.from_function(
        func=run, name=tool.name, description=tool.description, args_schema=tool.Args,
        response_format="content_and_artifact", handle_tool_error=True,
    )


def langchain_tools(executor: ToolExecutor, names: list[str] | None = None) -> list[StructuredTool]:
    selected = names or list(executor.tools)
    return [to_langchain_tool(executor.tools[n], executor) for n in selected if n in executor.tools]
