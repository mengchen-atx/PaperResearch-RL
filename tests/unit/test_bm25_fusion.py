import math

import pytest

from atlas.retrieval.bm25 import BM25Index, tokenize
from atlas.retrieval.fusion import min_max, reciprocal_rank_fusion, weighted_fusion

DOCS = ["the transformer uses self attention", "bm25 is a ranking function", "attention attention attention",
        "cats and dogs"]


def test_tokenize_drops_stopwords_keeps_numbers():
    assert tokenize("The BLEU of 28.4 is high") == ["bleu", "28.4", "high"]


def test_bm25_ranks_matching_docs_and_zero_for_nonmatching():
    idx = BM25Index().build(DOCS)
    s = idx.scores("attention")
    assert s[3] == 0 and s[1] == 0
    assert s[2] > 0 and s[0] > 0


def test_bm25_idf_formula():
    idx = BM25Index().build(DOCS)
    n, df = 4, 2
    assert idx.idf["attention"] == pytest.approx(math.log((n - df + 0.5) / (df + 0.5) + 1))


def test_bm25_roundtrip_serialization():
    idx = BM25Index().build(DOCS)
    idx2 = BM25Index.from_dict(idx.to_dict())
    assert (idx.scores("ranking attention") == idx2.scores("ranking attention")).all()


def test_bm25_empty_index_and_query():
    assert BM25Index().build([]).scores("x").size == 0
    assert BM25Index().build(DOCS).scores("the of").sum() == 0


def test_rrf_matches_formula():
    fused = reciprocal_rank_fusion({"a": [1, 2], "b": [2, 3]}, k=60)
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1] == pytest.approx(1 / 61)
    assert max(fused, key=fused.get) == 2


def test_min_max_and_weighted_fusion():
    assert min_max({1: 5.0, 2: 5.0}) == {1: 1.0, 2: 1.0}
    fused = weighted_fusion({1: 0.9, 2: 0.1}, {2: 10.0, 3: 0.0}, dense_weight=0.5)
    assert fused[1] == pytest.approx(0.5) and fused[2] == pytest.approx(0.5) and fused[3] == 0.0
