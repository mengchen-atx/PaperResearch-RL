"""Second-stage reranking.

First-stage retrievers are *bi-encoders* (query and chunk embedded separately),
which is fast but lossy: the model never sees query and chunk together. A
*cross-encoder* reads "[query] [SEP] [chunk]" jointly and outputs a relevance
score, which is much more accurate but costs one forward pass per candidate.

Hence the classic two-stage design: cheap retrieval over the whole corpus to get
`candidate_k` (~30) candidates, then an expensive reranker over only those, then
keep `top_k` (~5). Default model: ms-marco-MiniLM-L-6-v2 (22M params, CPU-friendly).
"""

from __future__ import annotations

from typing import Protocol

from atlas.common.config import RerankerConfig
from atlas.common.device import inference_lock, resolve_device
from atlas.common.types import ScoredChunk
from atlas.observability.logging import get_logger
from atlas.retrieval.bm25 import tokenize

log = get_logger("reranker")


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


def _with_score(sc: ScoredChunk, score: float, name: str) -> ScoredChunk:
    return sc.model_copy(
        update={"score": score, "retriever": f"{sc.retriever}+{name}",
                "component_scores": {**sc.component_scores, name: score}}
    )


class CrossEncoderReranker:
    name = "cross_encoder"

    def __init__(self, cfg: RerankerConfig) -> None:
        from sentence_transformers import CrossEncoder

        device = resolve_device(cfg.device)
        log.info("loading reranker", extra={"model": cfg.model_name, "device": device})
        self._model = CrossEncoder(cfg.model_name, device=device)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        with inference_lock():
            scores = self._model.predict([(query, c.chunk.text) for c in candidates], show_progress_bar=False)
        rescored = [_with_score(c, float(s), self.name) for c, s in zip(candidates, scores, strict=True)]
        return sorted(rescored, key=lambda c: c.score, reverse=True)[:top_k]


class LexicalOverlapReranker:
    """Model-free reranker (query-term coverage). Used in tests and as a cheap baseline."""

    name = "lexical"

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        q = set(tokenize(query))
        if not q:
            return candidates[:top_k]
        rescored = [
            _with_score(c, len(q & set(tokenize(c.chunk.text))) / len(q), self.name) for c in candidates
        ]
        return sorted(rescored, key=lambda c: c.score, reverse=True)[:top_k]


def build_reranker(cfg: RerankerConfig) -> Reranker | None:
    if not cfg.enabled:
        return None
    return CrossEncoderReranker(cfg) if cfg.provider == "cross_encoder" else LexicalOverlapReranker()
