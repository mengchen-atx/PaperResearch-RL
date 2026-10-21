"""Run one or more experiment configs against a benchmark and write results.

    python scripts/run_benchmark.py configs/experiments/hybrid_rag.yaml [more.yaml ...]
        [--benchmark data/benchmarks/atlas_papers_v1.jsonl] [--limit N] [--types table,figure]
        [--trajectories]   # agent runs: also append evaluated trajectories (for SFT/RL)

Results: benchmarks/results/<experiment>/{config.json,records.jsonl,summary.json}
Then:    python scripts/make_results_table.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atlas.common.config import load_config
from atlas.evaluation.runner import run_experiment
from atlas.observability.logging import configure_logging

HEADLINE = ["correct", "token_f1", "groundedness", "citation_precision", "citation_validity", "recall@5", "mrr",
            "tool_calls", "latency_p50_ms", "latency_p95_ms"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--benchmark", default="data/benchmarks/atlas_papers_v1.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--types", type=lambda s: s.split(","))
    ap.add_argument("--trajectories", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    configure_logging(cfg.observability.log_level)
    out = Path(args.out) if args.out else cfg.resolve_path(cfg.paths.runs_dir)
    for c in args.configs:
        s = run_experiment(Path(c), Path(args.benchmark), out, args.limit, args.types, args.trajectories)
        m = s["metrics"]
        print(f"\n== {s['experiment']['name']} ({s['num_items']} items, {s['wall_time_s']:.0f}s)")
        print(json.dumps({k: round(m[k], 4) for k in HEADLINE if k in m}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
