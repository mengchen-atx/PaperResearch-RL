# Design: Serving, async work, and deployment

## Single node (verified)
`atlas serve` → FastAPI + in-process job workers (memory queue) + FAISS index + local or remote LLM.
`atlas serve-model` → OpenAI-compatible HF-transformers server (vLLM stand-in on non-NVIDIA machines).

## Distributed (configured, not validated — no GPU/cluster available; manifests statically checked)
`docker/docker-compose.yml` (api, 2 workers, redis, qdrant, vLLM under the `gpu` profile) and
`deployment/kubernetes/` (Deployments/StatefulSets with requests/limits and probes, HPA for the API, KEDA
ScaledObject for workers on Redis queue depth, vLLM with 1 GPU/replica, RWX volume for the index, ingress with
rate limiting and SSE-friendly timeouts, secrets via Secret objects).

## vLLM notes
> Status: configured, not validated (no GPU available). No vLLM numbers were measured.

* `--gpu-memory-utilization 0.9`: fraction of VRAM for weights + KV cache; the remainder is headroom.
* KV cache: PagedAttention allocates KV in blocks, so memory scales with *actual* tokens in flight; max concurrency
  is bounded by `--max-num-seqs` and free KV blocks. Long evidence prompts (~1.2k tokens here) × concurrent
  sequences is the capacity planning unit.
* `--enable-prefix-caching`: agent prompts reuse long identical prefixes (system prompt + evidence across router
  turns) → prefill is skipped for cached blocks, reducing TTFT.
* Quantization: AWQ/GPTQ checkpoints (`--quantization awq`) roughly halve memory vs fp16 with small quality cost;
  FP8 on Hopper/Ada.
* Tensor parallelism (`--tensor-parallel-size N`) for models that don't fit one GPU; otherwise scale replicas.
* Batching: continuous batching is automatic; throughput rises with concurrency until KV blocks run out.

Measure with `scripts/bench_inference.py` (TTFT via streaming, decode tok/s, req/s at a given concurrency).

## Async jobs
202 + job id; `GET /research/{id}` and SSE `GET /research/{id}/events`. Redis backend: job JSON with TTL, event
lists, `LPUSH/BRPOP` work queue, `SET NX` idempotency keys. Delivery is at-least-once; handlers skip finished jobs;
failed attempts re-queue up to a limit; workers drain gracefully on SIGTERM.
