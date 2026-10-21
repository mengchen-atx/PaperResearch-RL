"""Shared fixtures: a small synthetic corpus and a fully offline Atlas runtime.

Offline means: hashing embedder (no model download), extractive LLM, fake VLM,
heuristic planner/critic. Tests exercise the real pipeline code paths without
network or GPUs; real-model runs are covered by `-m slow` tests and benchmarks.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from atlas.common.config import AtlasConfig
from atlas.common.types import TokenUsage
from atlas.multimodal.vlm import VLMResponse

WIDGET_TEXT = (
    "The Atlas widget is a retrieval component. It was introduced in 2024 by the Atlas team. "
    "Self-attention relates different positions of a single sequence to compute a representation. "
    "The widget uses 8 attention heads and a hidden size of 512."
)


def make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    # page 1: prose
    p1 = doc.new_page()
    p1.insert_textbox(pymupdf.Rect(50, 50, 550, 400), "1 Introduction\n" + WIDGET_TEXT * 2, fontsize=11)
    # page 2: a ruled table with a caption
    p2 = doc.new_page()
    p2.insert_text((50, 60), "Table 1: Widget benchmark results on the Foo dataset.", fontsize=11)
    rows = [["Model", "BLEU", "Params"], ["Widget-S", "25.1", "60"], ["Widget-L", "28.4", "210"],
            ["Baseline", "24.0", "55"]]
    x0, y0, cw, rh = 50, 80, 150, 22
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            p2.draw_rect(rect, color=(0, 0, 0), width=0.8)
            p2.insert_text((rect.x0 + 4, rect.y1 - 7), cell, fontsize=10)
    # page 3: a "chart" (drawn bars) above a figure caption
    p3 = doc.new_page()
    for i, h in enumerate([60, 120, 90]):
        p3.draw_rect(pymupdf.Rect(100 + i * 80, 300 - h, 150 + i * 80, 300), color=(0, 0, 1), fill=(0.3, 0.5, 0.9))
    p3.insert_text((50, 330), "Figure 1: Bar chart of widget accuracy across three datasets.", fontsize=11)
    doc.save(path)
    doc.close()


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("corpus")
    make_pdf(d / "widget_paper.pdf")
    (d / "notes.md").write_text(
        "# Background\nBM25 is a sparse lexical ranking function used in search engines.\n\n"
        "# Dense retrieval\nDense retrievers embed queries and documents into vectors and compare them "
        "with cosine similarity.\n")
    (d / "facts.txt").write_text("The capital of the fictional country Zedland is Quorra. Quorra has 3 rivers.")
    return d


def offline_config(root: Path, **overrides) -> AtlasConfig:
    base = {
        "paths": {"data_dir": str(root / "data"), "index_dir": str(root / "index"),
                  "trajectories_dir": str(root / "traj"), "runs_dir": str(root / "runs"),
                  "cache_dir": str(root / "cache"), "trace_dir": str(root / "traces"),
                  "upload_dir": str(root / "data" / "raw" / "uploads")},
        "chunking": {"chunk_size": 300, "chunk_overlap": 50, "min_chunk_chars": 20},
        "embedding": {"provider": "hashing", "hashing_dim": 256, "cache": False},
        "retrieval": {"mode": "hybrid", "vector_store": "numpy", "top_k": 4, "candidate_k": 10},
        "reranker": {"enabled": False},
        "llm": {"provider": "extractive", "model": "extractive"},
        "vlm": {"provider": "none"},
        "agent": {"policy": "heuristic", "planner": "heuristic", "critic": "heuristic", "max_iterations": 5,
                  "max_tool_calls": 6, "timeout_s": 60, "min_evidence": 1, "sufficient_score": 0.5},
        "tools": {"python_mode": "sandbox", "sql_db_path": str(root / "index" / "tables.sqlite"),
                  "web_backend": "disabled"},
        "cache": {"backend": "memory"},
        "serving": {"queue_backend": "memory", "max_concurrent_jobs": 1, "rate_limit_per_minute": 1000},
    }
    return AtlasConfig().with_overrides(base).with_overrides(overrides)


class FakeVLM:
    model = "fake-vlm"

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, image_path: Path, prompt: str) -> VLMResponse:
        self.calls += 1
        return VLMResponse(text="The bar chart shows accuracy rising to a peak on the second dataset.",
                           model=self.model, latency_ms=1.0, usage=TokenUsage())


@pytest.fixture(scope="session")
def built_runtime(tmp_path_factory, corpus_dir):
    """Runtime with an index built from the synthetic corpus (session-scoped: built once)."""
    from atlas.runtime import AtlasRuntime

    root = tmp_path_factory.mktemp("atlas")
    rt = AtlasRuntime(offline_config(root))
    report = rt.ingest([corpus_dir])
    assert report.documents == 3
    return rt


@pytest.fixture
def runtime_factory(tmp_path, corpus_dir):
    from atlas.runtime import AtlasRuntime

    def make(**overrides):
        rt = AtlasRuntime(offline_config(tmp_path, **overrides))
        rt.ingest([corpus_dir])
        return rt

    return make
