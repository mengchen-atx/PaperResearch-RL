"""Inference benchmark for any OpenAI-compatible endpoint (vLLM, Ollama, atlas dev server).

Measures, per request: time-to-first-token (TTFT, via streaming), end-to-end latency,
output tokens/sec; and aggregate throughput (requests/s, output tokens/s) at a given
concurrency. Run against vLLM on a GPU host to measure continuous-batching gains,
or against `atlas serve-model` locally as a single-request baseline.

    python scripts/bench_inference.py --base-url http://localhost:8001/v1 --model <id> \
        --requests 16 --concurrency 4 --max-tokens 128 --out benchmarks/results/inference/<name>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from pathlib import Path

import httpx
import numpy as np

PROMPTS = [
    "Explain scaled dot-product attention in three sentences.",
    "What is the difference between BM25 and dense retrieval?",
    "Summarize the idea of retrieval-augmented generation.",
    "Why do larger language models tend to be more sample efficient?",
]


async def one_request(client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True}
    t0 = time.perf_counter()
    ttft, pieces = None, []
    async with client.stream("POST", "/chat/completions", json=payload) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            delta = json.loads(body)["choices"][0].get("delta", {}).get("content")
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                pieces.append(delta)
    total = time.perf_counter() - t0
    # Token count approximated by streamed chunks (vLLM/dev server emit ~1 token per chunk).
    n_out = len(pieces)
    return {"ttft_s": ttft or total, "latency_s": total, "output_tokens": n_out,
            "tokens_per_s": n_out / max(total - (ttft or 0), 1e-6)}


async def run(args) -> dict:
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as client:
        async def guarded(i):
            async with sem:
                return await one_request(client, args.model, PROMPTS[i % len(PROMPTS)], args.max_tokens)

        await guarded(0)  # warm-up (excluded)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[guarded(i) for i in range(args.requests)])
        wall = time.perf_counter() - t0
    col = lambda k: [r[k] for r in results]  # noqa: E731
    return {
        "endpoint": args.base_url, "model": args.model, "requests": args.requests, "concurrency": args.concurrency,
        "max_tokens": args.max_tokens, "wall_s": wall,
        "ttft_p50_s": float(np.percentile(col("ttft_s"), 50)), "ttft_p95_s": float(np.percentile(col("ttft_s"), 95)),
        "latency_p50_s": float(np.percentile(col("latency_s"), 50)),
        "latency_p95_s": float(np.percentile(col("latency_s"), 95)),
        "decode_tokens_per_s_mean": float(np.mean(col("tokens_per_s"))),
        "throughput_req_per_s": args.requests / wall, "throughput_output_tokens_per_s": sum(col("output_tokens")) / wall,
        "host": platform.platform(), "label": args.label,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    args = ap.parse_args()
    res = asyncio.run(run(args))
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
