"""Evaluation metrics (pure functions, unit-tested).

Retrieval (page-level relevance against gold evidence units):
    Recall@K     fraction of gold evidence units hit by the top-K results
    Precision@K  fraction of the top-K results that are relevant
    MRR          1 / rank of the first relevant result
    nDCG@K       binary-gain DCG normalized by the ideal ordering

Generation:
    answer correctness  numeric match within tolerance (numeric items) else
                        normalized containment of the gold answer or an alias
    token F1            SQuAD-style token overlap with the gold answer
    groundedness        share of answer sentences whose content words are mostly
                        present in the evidence (a lexical support proxy — not an
                        NLI/LLM judgment; reported as such)
Citations:
    validity   valid markers / all markers (invalid = pointing at non-existent evidence)
    precision  cited evidence that is gold-relevant / cited evidence
    coverage   answer sentences carrying >=1 citation / answer sentences
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Sequence

import numpy as np

from atlas.evaluation.benchmark import BenchmarkItem, GoldEvidence
from atlas.rag.citations import extract_markers, split_claims
from atlas.retrieval.bm25 import tokenize

# ---------------------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------------------
Location = tuple[str | None, int | None]  # (filename, page)


def _relevant(loc: Location, gold: Sequence[GoldEvidence]) -> bool:
    return any(g.matches(loc[0], loc[1]) for g in gold)


def recall_at_k(retrieved: Sequence[Location], gold: Sequence[GoldEvidence], k: int) -> float:
    if not gold:
        return 0.0
    top = retrieved[:k]
    hit = sum(1 for g in gold if any(g.matches(f, p) for f, p in top))
    return hit / len(gold)


def precision_at_k(retrieved: Sequence[Location], gold: Sequence[GoldEvidence], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for loc in retrieved[:k] if _relevant(loc, gold)) / k


def reciprocal_rank(retrieved: Sequence[Location], gold: Sequence[GoldEvidence]) -> float:
    for rank, loc in enumerate(retrieved, start=1):
        if _relevant(loc, gold):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[Location], gold: Sequence[GoldEvidence], k: int) -> float:
    """Binary nDCG where each gold evidence unit is credited once (the first time it is hit).

    Several chunks can come from the same gold page; crediting each would let DCG exceed
    the ideal DCG. With novelty-aware gains, the ideal ranking hits one new unit per rank,
    so IDCG = sum_{i < min(k, |gold|)} 1/log2(i+2) and nDCG is in [0, 1].
    """
    hit: set[int] = set()
    dcg = 0.0
    for i, (f, p) in enumerate(retrieved[:k]):
        new = [j for j, g in enumerate(gold) if j not in hit and g.matches(f, p)]
        if new:
            hit.add(new[0])
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(k, len(gold))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------------------
_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def normalize_answer(text: str) -> str:
    text = text.lower().replace("−", "-")
    text = "".join(ch for ch in text if ch not in set(string.punctuation) - {".", "%", "/", "√", "-"})
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_answer(prediction: str, answers: Sequence[str]) -> bool:
    pred = normalize_answer(prediction)
    for ans in answers:
        a = normalize_answer(ans)
        if not a:
            continue
        # numbers/short tokens must match on token boundaries ("4" must not match "40")
        pattern = r"(?<![\w.])" + re.escape(a) + r"(?![\w]|\.\d)"
        if re.search(pattern, pred):
            return True
    return False


def extract_numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUM.findall(text)]


def numeric_match(prediction: str, target: float, tolerance: float) -> bool:
    return any(abs(n - target) <= tolerance for n in extract_numbers(prediction))


def token_f1(prediction: str, gold: str) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if not p or not g or overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def answer_correct(prediction: str, item: BenchmarkItem) -> bool:
    if item.numeric is not None:
        return numeric_match(prediction, item.numeric, item.tolerance or 0.01)
    return contains_answer(prediction, [item.answer, *item.aliases])


# ---------------------------------------------------------------------------------------
# Groundedness & citations
# ---------------------------------------------------------------------------------------
SUPPORT_THRESHOLD = 0.6
_MARKER_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


def groundedness(answer: str, evidence_texts: Sequence[str], threshold: float = SUPPORT_THRESHOLD) -> float:
    claims = [c for c in split_claims(_MARKER_RE.sub("", answer)) if tokenize(c)]
    if not claims:
        return 0.0
    pool: set[str] = set()
    for t in evidence_texts:
        pool |= set(tokenize(t))
    supported = 0
    for c in claims:
        toks = set(tokenize(c))
        if toks and len(toks & pool) / len(toks) >= threshold:
            supported += 1
    return supported / len(claims)


def citation_metrics(answer: str, evidence_locs: Sequence[Location],
                     gold: Sequence[GoldEvidence]) -> dict[str, float | None]:
    """validity is undefined (None) without markers; precision is undefined without valid markers.
    Keeping these as None (not 0) avoids conflating 'did not cite' with 'cited wrongly'."""
    markers = extract_markers(answer)
    valid = [m for m in markers if 1 <= m <= len(evidence_locs)]
    claims = split_claims(answer)
    cited_claims = sum(1 for c in claims if _MARKER_RE.search(c))
    rel = [m for m in valid if _relevant(evidence_locs[m - 1], gold)]
    return {
        "citation_count": float(len(markers)),
        "citation_validity": len(valid) / len(markers) if markers else None,
        "citation_precision": len(rel) / len(valid) if valid else None,
        "citation_coverage": cited_claims / len(claims) if claims else 0.0,
        "has_valid_citation": float(bool(valid)),
    }


# ---------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------
def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0
