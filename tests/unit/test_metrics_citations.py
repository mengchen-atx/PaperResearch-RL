import pytest

from atlas.common.types import Evidence, Modality
from atlas.evaluation import metrics as M
from atlas.evaluation.benchmark import BenchmarkItem, GoldEvidence
from atlas.rag.citations import extract_markers, resolve_citations
from atlas.rag.prompts import format_evidence, neutralize_source_refs

GOLD = [GoldEvidence(filename="a.pdf", pages=[2]), GoldEvidence(filename="b.pdf", pages=[5])]
RETRIEVED = [("a.pdf", 1), ("a.pdf", 2), ("c.pdf", 3), ("b.pdf", 5)]


def test_retrieval_metrics_hand_computed():
    assert M.recall_at_k(RETRIEVED, GOLD, 2) == 0.5
    assert M.recall_at_k(RETRIEVED, GOLD, 4) == 1.0
    assert M.precision_at_k(RETRIEVED, GOLD, 4) == 0.5
    assert M.reciprocal_rank(RETRIEVED, GOLD) == 0.5
    dcg = 1 / 1.5849625 + 1 / 2.3219281  # ranks 2 and 4
    idcg = 1 + 1 / 1.5849625
    assert M.ndcg_at_k(RETRIEVED, GOLD, 4) == pytest.approx(dcg / idcg, rel=1e-4)
    assert M.reciprocal_rank([("z", 1)], GOLD) == 0.0
    # many chunks from the same gold page must not push nDCG above 1
    assert M.ndcg_at_k([("a.pdf", 2)] * 5, GOLD, 5) <= 1.0
    assert M.ndcg_at_k([("a.pdf", 2), ("b.pdf", 5)], GOLD, 5) == pytest.approx(1.0)


def test_contains_answer_respects_token_boundaries():
    assert M.contains_answer("The model uses 8 heads.", ["8"])
    assert not M.contains_answer("The model uses 80 heads.", ["8"])
    assert not M.contains_answer("score of 28.45", ["28.4"])
    assert M.contains_answer("BLEU of 28.4.", ["28.4"])
    assert M.contains_answer("It is BART-large based", ["BART-large"])


def test_numeric_match_and_f1():
    assert M.numeric_match("the gap is 2.04 BLEU", 2.04, 0.02)
    assert M.numeric_match("about 2,500 million", 2500, 1)
    assert not M.numeric_match("no numbers", 1.0, 0.1)
    assert M.token_f1("sine and cosine functions", "sine and cosine") == pytest.approx(0.857, rel=1e-2)


def test_answer_correct_numeric_vs_text():
    item = BenchmarkItem(id="x", question="q", answer="2.04", numeric=2.04, tolerance=0.02)
    assert M.answer_correct("difference = 2.05", item)
    item2 = BenchmarkItem(id="y", question="q", answer="DPR", aliases=["Dense Passage Retriever"])
    assert M.answer_correct("It uses the dense passage retriever.", item2)


def test_groundedness_and_citation_metrics():
    ev = ["The Transformer uses eight attention heads in each layer."]
    assert M.groundedness("The Transformer uses eight attention heads [1].", ev) == 1.0
    assert M.groundedness("Cats purr loudly at night.", ev) == 0.0
    cm = M.citation_metrics("Claim one [1]. Claim two [3]. Claim three.", [("a.pdf", 2), ("x.pdf", 1)], GOLD)
    assert cm["citation_validity"] == 0.5  # [3] points at non-existent evidence
    assert cm["citation_precision"] == 1.0
    assert cm["citation_coverage"] == pytest.approx(2 / 3)
    none = M.citation_metrics("No citations here.", [("a.pdf", 2)], GOLD)
    assert none["citation_validity"] is None and none["citation_precision"] is None
    assert none["has_valid_citation"] == 0.0


def test_extract_and_resolve_citations():
    assert extract_markers("a [1] b [2, 3] c [1][4]") == [1, 2, 3, 4]
    ev = [Evidence(evidence_id="e1", modality=Modality.TEXT, content="c1", source_tool="t", filename="a.pdf", page=1)]
    valid, invalid = resolve_citations("x [1] y [7]", ev)
    assert [c.evidence_id for c in valid] == ["e1"] and invalid == [7]


def test_source_refs_are_neutralized_in_context():
    assert neutralize_source_refs("as in [34, 2, 9] and [5]") == "as in (ref 34, 2, 9) and (ref 5)"
    ev = [Evidence(evidence_id="e", modality=Modality.TEXT, content="see [12]", source_tool="t", filename="f")]
    assert "[12]" not in format_evidence(ev) and format_evidence(ev).startswith("[1] (f)")
