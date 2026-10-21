"""System benchmark: load-test the running Atlas API (/retrieve and /ask) at fixed concurrency.

    atlas serve &   # in another terminal
    python scripts/run_system_benchmark.py --base-url http://127.0.0.1:8000 --endpoint retrieve \
        --requests 200 --concurrency 8 --out benchmarks/results/system/retrieve_c8.json
Reports p50/p95/p99 latency, error rate and throughput (req/s) from real HTTP calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

from atlas.evaluation.benchmark import load_benchmark


async def main_async(args) -> dict:
    questions = [i.question for i in load_benchmark(Path(args.benchmark))]
    sem = asyncio.Semaphore(args.concurrency)
    lat, errors = [], 0
    async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as client:
        async def call(i):
            nonlocal errors
            q = questions[i % len(questions)]
            body = {"query": q, "top_k": 5} if args.endpoint == "retrieve" else {"question": q}
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(f"/{args.endpoint}", json=body)
                    r.raise_for_status()
                    lat.append(time.perf_counter() - t0)
                except httpx.HTTPError:
                    errors += 1

        t0 = time.perf_counter()
        await asyncio.gather(*[call(i) for i in range(args.requests)])
        wall = time.perf_counter() - t0
    return {"endpoint": args.endpoint, "requests": args.requests, "concurrency": args.concurrency,
            "errors": errors, "wall_s": wall, "throughput_rps": len(lat) / wall,
            "latency_p50_ms": float(np.percentile(lat, 50) * 1000) if lat else None,
            "latency_p95_ms": float(np.percentile(lat, 95) * 1000) if lat else None,
            "latency_p99_ms": float(np.percentile(lat, 99) * 1000) if lat else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--endpoint", choices=["retrieve", "ask"], default="retrieve")
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--benchmark", default="data/benchmarks/atlas_papers_v1.jsonl")
    ap.add_argument("--out")
    args = ap.parse_args()
    res = asyncio.run(main_async(args))
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
