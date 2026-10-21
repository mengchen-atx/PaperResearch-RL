"""Benchmark schema.

Each item has a question, a gold short answer (+ aliases, optional numeric value
with tolerance), a type (text/table/compute/figure/multihop), the modalities the
question needs, and gold evidence as (filename, pages, quote). Gold quotes are
verified against the PDFs by `scripts/build_benchmark.py`.

Relevance for retrieval metrics is page-level: a retrieved item is relevant to
a gold evidence unit if it comes from the same file and one of its pages.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldEvidence(BaseModel):
    filename: str
    pages: list[int]
    quote: str = ""

    def matches(self, filename: str | None, page: int | None) -> bool:
        return filename == self.filename and page in self.pages


class BenchmarkItem(BaseModel):
    id: str
    question: str
    answer: str
    aliases: list[str] = Field(default_factory=list)
    type: str = "text"
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    evidence: list[GoldEvidence] = Field(default_factory=list)
    numeric: float | None = None
    tolerance: float | None = None


def load_benchmark(path: Path, types: list[str] | None = None, limit: int | None = None) -> list[BenchmarkItem]:
    items = [BenchmarkItem.model_validate(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    if types:
        items = [i for i in items if i.type in types]
    return items[:limit] if limit else items


def split_items(items: list[BenchmarkItem], test_fraction: float = 0.3, seed: int = 0):
    """Deterministic train/test split (used by SFT and RL to avoid evaluating on training questions)."""
    import random

    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction))
    return shuffled[n_test:], shuffled[:n_test]
