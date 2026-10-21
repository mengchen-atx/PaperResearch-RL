"""LangChain adapters: Atlas retriever/tools behave as standard LangChain components."""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from atlas.common.types import Modality
from atlas.integrations.langchain import AtlasRetriever, langchain_tools
from atlas.retrieval.filters import MetadataFilter


def test_retriever_returns_documents_with_citation_metadata(built_runtime):
    docs = AtlasRetriever(atlas_retriever=built_runtime.retriever, top_k=3).invoke("capital of Zedland")
    assert docs and all(isinstance(d, Document) for d in docs) and len(docs) <= 3
    top = docs[0]
    assert top.metadata["filename"] == "facts.txt" and "Quorra" in top.page_content
    assert top.metadata["citation"].startswith("facts.txt") and top.id == top.metadata["chunk_id"]


def test_retriever_respects_metadata_filter(built_runtime):
    flt = MetadataFilter(modalities=[Modality.TABLE])
    docs = AtlasRetriever(atlas_retriever=built_runtime.retriever, metadata_filter=flt).invoke("widget results")
    assert docs and {d.metadata["modality"] for d in docs} == {"table"}


def test_tools_expose_atlas_schemas(built_runtime):
    tools = {t.name: t for t in langchain_tools(built_runtime.executor)}
    assert {"search_documents", "lookup_table", "query_sql", "run_python"} <= set(tools)
    assert "query" in tools["search_documents"].args


def test_tool_call_returns_content_and_structured_artifact(built_runtime):
    tool = {t.name: t for t in langchain_tools(built_runtime.executor)}["search_documents"]
    msg = tool.invoke({"type": "tool_call", "id": "call_1", "name": tool.name,
                       "args": {"query": "capital of Zedland", "top_k": 2}})
    assert isinstance(msg, ToolMessage) and "Quorra" in msg.content
    assert msg.artifact and msg.artifact[0]["filename"] == "facts.txt"


def test_tool_errors_are_returned_not_raised(built_runtime):
    tool = {t.name: t for t in langchain_tools(built_runtime.executor)}["query_sql"]
    out = tool.invoke({"sql": "DROP TABLE atlas_tables"})
    assert "Only SELECT" in out


def test_invalid_arguments_are_rejected(built_runtime):
    tool = {t.name: t for t in langchain_tools(built_runtime.executor)}["search_documents"]
    with pytest.raises(ValidationError):
        tool.invoke({"query": "x", "top_k": 999})
