"""Retrievers: dense, BM25, hybrid (+ optional rerank and query rewriting).

    query
      -> [query rewrite]                      (optional)
      -> BM25 top-C   and   dense top-C       (C = candidate_k)
      -> fusion (RRF or weighted)             (hybrid only)
      -> [cross-encoder rerank of candidates] (optional)
      -> top-K ScoredChunks

`Retriever.retrieve` is the single entry point used by RAG pipelines and the
agent's `search_documents` tool, so every ablation differs only in config.
"""

from __future__ import annotations

import time

from atlas.common.cache import Cache, NullCache, make_key
from atlas.common.config import RetrievalConfig
from atlas.common.types import ScoredChunk
from atlas.embeddings.embedder import Embedder
from atlas.observability.metrics import RETRIEVAL_LATENCY
from atlas.observability.tracing import get_tracer
from atlas.reranking.reranker import Reranker
from atlas.retrieval.bm25 import tokenize
from atlas.retrieval.filters import MetadataFilter, allowed_indices
from atlas.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from atlas.retrieval.index import KnowledgeIndex
from atlas.retrieval.vector_store import top_k_from_scores


def keyword_rewrite(query: str) -> str:
    """Cheap query rewrite: drop stopwords/question scaffolding, keep content terms.

    Helps BM25 on verbose natural-language questions. An LLM-based rewrite
    (`query_rewrite: llm`) is supported by the RAG pipeline for comparison.
    """
    terms = tokenize(query)
    return " ".join(terms) if terms else query


class Retriever:
    def __init__(
        self,
        index: KnowledgeIndex,
        embedder: Embedder,
        cfg: RetrievalConfig,
        reranker: Reranker | None = None,
        cache: Cache | None = None,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.cfg = cfg
        self.reranker = reranker
        self.cache = cache or NullCache()

    @property
    def signature(self) -> dict:
        return {
            "mode": self.cfg.mode, "fusion": self.cfg.fusion, "rerank": self.reranker.name if self.reranker else None,
            "store": self.index.manifest.get("vector_store"), "emb": self.embedder.name,
            "built": self.index.manifest.get("built_at"),
        }

    # --- first-stage retrievers -------------------------------------------------
    def dense(self, query: str, k: int, flt: MetadataFilter | None = None) -> list[tuple[int, float]]:
        qv = self.embedder.embed_query(query)
        allowed = allowed_indices(self.index.chunks, flt)
        return self.index.vector_store.search(qv, k, allowed=allowed, flt=flt)

    def sparse(self, query: str, k: int, flt: MetadataFilter | None = None) -> list[tuple[int, float]]:
        scores = self.index.bm25.scores(query)
        allowed = allowed_indices(self.index.chunks, flt)
        return [(i, s) for i, s in top_k_from_scores(scores, k, allowed) if s > 0]

    # --- full pipeline ------------------------------------------------------------
    def retrieve(
        self, query: str, top_k: int | None = None, flt: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        k = top_k or self.cfg.top_k
        key = make_key("retrieve", {"q": query, "k": k, "f": flt.model_dump() if flt else None, **self.signature})
        cached = self.cache.get(key)
        if cached is not None:
            return [ScoredChunk.model_validate(x) for x in cached]

        t0 = time.perf_counter()
        with get_tracer().span("retrieve", kind="retrieval", mode=self.cfg.mode, k=k) as attrs:
            q = keyword_rewrite(query) if self.cfg.query_rewrite == "keywords" else query
            results = self._retrieve_uncached(query, q, k, flt)
            attrs["num_results"] = len(results)
            attrs["top_score"] = results[0].score if results else None
        RETRIEVAL_LATENCY.labels(self.cfg.mode).observe(time.perf_counter() - t0)
        self.cache.set(key, [r.model_dump(mode="json") for r in results])
        return results

    def _retrieve_uncached(self, original: str, q: str, k: int, flt: MetadataFilter | None) -> list[ScoredChunk]:
        # With a reranker we over-fetch candidates, otherwise fetch exactly k.
        n_cand = max(self.cfg.candidate_k, k) if (self.reranker or self.cfg.mode == "hybrid") else k
        chunks = self.index.chunks

        if self.cfg.mode == "dense":
            hits = self.dense(q, n_cand, flt)
            cands = [ScoredChunk(chunk=chunks[i], score=s, retriever="dense", component_scores={"dense": s})
                     for i, s in hits]
        elif self.cfg.mode == "bm25":
            hits = self.sparse(q, n_cand, flt)
            cands = [ScoredChunk(chunk=chunks[i], score=s, retriever="bm25", component_scores={"bm25": s})
                     for i, s in hits]
        else:
            dense_hits = dict(self.dense(q, n_cand, flt))
            sparse_hits = dict(self.sparse(q, n_cand, flt))
            if self.cfg.fusion == "rrf":
                fused = reciprocal_rank_fusion(
                    {"dense": list(dense_hits), "bm25": list(sparse_hits)}, k=self.cfg.rrf_k
                )
            else:
                fused = weighted_fusion(dense_hits, sparse_hits, self.cfg.dense_weight)
            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:n_cand]
            cands = [
                ScoredChunk(
                    chunk=chunks[i], score=s, retriever=f"hybrid-{self.cfg.fusion}",
                    component_scores={"fused": s, **({"dense": dense_hits[i]} if i in dense_hits else {}),
                                      **({"bm25": sparse_hits[i]} if i in sparse_hits else {})},
                )
                for i, s in ranked
            ]

        if self.reranker:
            # Rerank against the user's original wording, not the keyword rewrite.
            return self.reranker.rerank(original, cands, k)
        return cands[:k]
