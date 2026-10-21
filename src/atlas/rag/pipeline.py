"""RAG pipelines — the non-agentic baselines of the ablation ladder.

    BaseLLMPipeline : question -> LLM (closed book, no retrieval)
    RAGPipeline     : question -> [rewrite] -> retrieve top-K -> prompt -> LLM -> cited answer

Vanilla / hybrid / hybrid+rerank RAG are all `RAGPipeline` with a different
`Retriever` config, so ablations change exactly one variable at a time.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from atlas.common.types import Citation, Evidence, ScoredChunk, TokenUsage
from atlas.llm.base import LLMClient
from atlas.rag.citations import resolve_citations
from atlas.rag.prompts import closed_book_messages, grounded_messages, rewrite_messages
from atlas.retrieval.filters import MetadataFilter
from atlas.retrieval.retriever import Retriever


class RAGAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    retrieved: list[ScoredChunk] = Field(default_factory=list)
    search_query: str = ""
    usage: TokenUsage = TokenUsage()
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    model: str = ""


class BaseLLMPipeline:
    name = "base_llm"

    def __init__(self, llm: LLMClient, max_new_tokens: int = 256) -> None:
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    def answer(self, question: str, **_: object) -> RAGAnswer:
        t0 = time.perf_counter()
        resp = self.llm.generate(closed_book_messages(question), max_tokens=self.max_new_tokens, purpose="answer")
        total = (time.perf_counter() - t0) * 1000
        return RAGAnswer(question=question, answer=resp.text, usage=resp.usage, generation_ms=total,
                         total_ms=total, model=resp.model)


class RAGPipeline:
    name = "rag"

    def __init__(self, retriever: Retriever, llm: LLMClient, max_new_tokens: int = 256,
                 llm_query_rewrite: bool = False) -> None:
        self.retriever = retriever
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.llm_query_rewrite = llm_query_rewrite

    def retrieve(self, question: str, top_k: int | None = None,
                 flt: MetadataFilter | None = None) -> tuple[list[ScoredChunk], str, TokenUsage]:
        query, usage = question, TokenUsage()
        if self.llm_query_rewrite:
            r = self.llm.generate(rewrite_messages(question), max_tokens=48, purpose="rewrite")
            query, usage = (r.text.strip().splitlines() or [question])[0] or question, r.usage
        return self.retriever.retrieve(query, top_k=top_k, flt=flt), query, usage

    def answer(self, question: str, top_k: int | None = None, flt: MetadataFilter | None = None) -> RAGAnswer:
        t0 = time.perf_counter()
        retrieved, query, usage = self.retrieve(question, top_k, flt)
        t_ret = time.perf_counter()
        evidence = [Evidence.from_scored_chunk(sc, "search_documents") for sc in retrieved]
        if not evidence:
            # Empty retrieval: do not let the model answer ungrounded.
            return RAGAnswer(question=question, answer="No relevant evidence was found in the knowledge base.",
                             search_query=query, usage=usage, retrieval_ms=(t_ret - t0) * 1000,
                             total_ms=(time.perf_counter() - t0) * 1000, model=self.llm.model)
        resp = self.llm.generate(grounded_messages(question, evidence), max_tokens=self.max_new_tokens,
                                 purpose="answer")
        citations, invalid = resolve_citations(resp.text, evidence)
        t_end = time.perf_counter()
        return RAGAnswer(
            question=question, answer=resp.text, citations=citations, invalid_citations=invalid,
            evidence=evidence, retrieved=retrieved, search_query=query, usage=usage + resp.usage,
            retrieval_ms=(t_ret - t0) * 1000, generation_ms=(t_end - t_ret) * 1000,
            total_ms=(t_end - t0) * 1000, model=resp.model,
        )
