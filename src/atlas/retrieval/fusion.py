"""Score fusion for hybrid retrieval.

BM25 scores are unbounded (0..~30) while cosine similarities live in [-1, 1], so
raw scores cannot simply be added. Two standard fixes:

  * Reciprocal Rank Fusion (RRF, Cormack et al. 2009):
        RRF(d) = sum_r 1 / (k + rank_r(d))
    Uses only ranks, so it is immune to score-scale differences. k=60 damps the
    influence of top ranks. Robust default; no tuning.

  * Weighted min-max fusion:
        s(d) = w * minmax(dense)(d) + (1 - w) * minmax(bm25)(d)
    Keeps score magnitudes (a clear winner stays a clear winner) but depends on
    the candidate set's score distribution and needs the weight w tuned.
"""

from __future__ import annotations


def reciprocal_rank_fusion(ranked_lists: dict[str, list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in ranked_lists.values():
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


def min_max(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {d: 1.0 for d in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores.items()}


def weighted_fusion(dense: dict[int, float], sparse: dict[int, float], dense_weight: float) -> dict[int, float]:
    nd, ns = min_max(dense), min_max(sparse)
    return {
        d: dense_weight * nd.get(d, 0.0) + (1 - dense_weight) * ns.get(d, 0.0)
        for d in set(nd) | set(ns)
    }
