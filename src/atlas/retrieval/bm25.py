"""Okapi BM25 sparse retrieval, implemented from scratch.

Why BM25 next to dense embeddings? Dense retrievers are great at paraphrase
("how many heads" ~ "number of attention heads") but weak on exact tokens that
matter in research: model names, numbers, acronyms ("BLEU", "WMT 2014", "d_k").
BM25 matches those exactly. Hybrid retrieval combines both failure profiles.

Score of document D for query Q:

    BM25(D, Q) = sum_{t in Q} IDF(t) * tf(t,D) * (k1 + 1)
                                  / (tf(t,D) + k1 * (1 - b + b * |D| / avgdl))
    IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

k1 controls term-frequency saturation; b controls document-length normalization.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

# Compact English stopword list; BM25 IDF already down-weights common words, this
# just keeps posting lists small.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "what", "which",
    "who", "with", "how", "does", "do", "did", "than", "then", "there", "these", "those", "into",
    "their", "our", "we", "you", "your", "can", "could", "would", "should", "will", "about",
})


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc_idx, tf)]
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def build(self, texts: list[str]) -> BM25Index:
        self.doc_len = []
        postings: dict[str, list[tuple[int, int]]] = {}
        for i, text in enumerate(texts):
            toks = tokenize(text)
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                postings.setdefault(term, []).append((i, tf))
        self.postings = postings
        n = len(texts)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {t: math.log((n - len(p) + 0.5) / (len(p) + 0.5) + 1.0) for t, p in postings.items()}
        return self

    def scores(self, query: str) -> np.ndarray:
        """Score every document (sparse accumulation over the query's posting lists)."""
        out = np.zeros(len(self.doc_len), dtype=np.float32)
        if not self.doc_len:
            return out
        dl = np.asarray(self.doc_len, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idx = np.fromiter((p[0] for p in plist), dtype=np.int64, count=len(plist))
            tf = np.fromiter((p[1] for p in plist), dtype=np.float32, count=len(plist))
            out[idx] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[idx])
        return out

    def to_dict(self) -> dict:
        return {"k1": self.k1, "b": self.b, "doc_len": self.doc_len, "postings": self.postings,
                "idf": self.idf, "avgdl": self.avgdl}

    @classmethod
    def from_dict(cls, d: dict) -> BM25Index:
        idx = cls(d["k1"], d["b"])
        idx.doc_len = d["doc_len"]
        idx.postings = {t: [tuple(p) for p in plist] for t, plist in d["postings"].items()}
        idx.idf = d["idf"]
        idx.avgdl = d["avgdl"]
        return idx
