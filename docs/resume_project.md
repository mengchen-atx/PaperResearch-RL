# Resume project entry: Atlas

Placeholders like `[XX%]` stay placeholders until a results file backs them (see `resume_metrics.md`).
The "Measured (current)" section at the bottom holds the only numbers that are backed today.

---

## Version A: LLM / GenAI Engineer

**Atlas: Multimodal Agentic Research System** · Python, PyTorch, Hugging Face (Transformers, PEFT, TRL), LangGraph, FAISS, Qdrant, FastAPI, Redis, Docker, Kubernetes

- Built an end-to-end research agent (LangGraph planner → router → tools → critic → writer) that answers questions over PDFs using hybrid retrieval, SQL over extracted tables, sandboxed Python, and a vision-language model, returning reports with verifiable page-level citations.
- Implemented RAG from scratch (sentence-aware chunking, bge embeddings, FAISS cosine search, hand-written BM25, RRF fusion, cross-encoder reranking). Raised Recall@5 from [XX%] to [XX%] and answer accuracy from [XX%] (closed-book) to [XX%] on a [N]-question page-labeled benchmark.
- Designed an evaluation harness covering retrieval (Recall@K, MRR, nDCG), answers, groundedness, citation validity/precision, and agent tool use, with gold labels auto-verified against source PDFs and bootstrap CIs on every ablation.
- Post-trained a [0.5B/1.5B/7B] open model on the agent's own filtered successful trajectories with LoRA/QLoRA (TRL + PEFT, [X%] of parameters trainable). Router action accuracy went from [XX%] to [XX%] on held-out trajectories.
- Formulated tool use as an MDP whose transitions are the production agent's own nodes. Trained the router with policy-gradient RL (modular reward: success, citation quality, potential-shaped evidence recall, tool cost), matching the baseline's success with [XX%] fewer tool calls; added a GRPO path for LLM routers.

## Version B: ML Engineer / ML Systems

**Atlas: Multimodal Agentic Research System** · PyTorch, vLLM, FastAPI, Redis, Qdrant, FAISS, Docker, Kubernetes, Prometheus

- Built a production-style serving stack: async FastAPI with SSE progress streaming, Redis job queue with idempotency keys, at-least-once delivery and bounded retries, horizontally scalable workers, rate limiting, request tracing, and Prometheus metrics.
- Kept inference behind a provider abstraction (local HF, vLLM/OpenAI-compatible, Anthropic). Benchmarked TTFT [XX ms], decode [XX tok/s], and throughput [XX req/s] at concurrency [N] on [hardware]. vLLM with prefix caching reached [X×] the throughput of the HF baseline.
- Load-tested the retrieval API at [XX req/s] with p95 [XX ms] under concurrency [N]. Cached retrieval, deterministic LLM responses, and embeddings.
- Wrote Kubernetes manifests: API HPA, queue-depth (KEDA) autoscaled agent workers, GPU vLLM deployment, Redis/Qdrant StatefulSets, probes and resource limits, and an ingress with rate limiting.
- Built a reproducible ML workflow: typed layered configs, experiment YAMLs that change one variable per ablation, machine-readable results, trajectory datasets with validation and filtering, [N] automated tests (unit/integration/e2e) running offline.

---

## Measured (current)

Numbers below are copied from `docs/experiments/results.md` (46-question benchmark, Apple M4 / MPS, Qwen2.5-1.5B-Instruct generator unless noted).

These bullets are backed today. Keep the qualifiers.

**LLM / GenAI Engineer**
- Built a multimodal research agent in Python and LangGraph (planner → router → tools → critic → writer). It answers questions over PDFs using hybrid retrieval, SQL over extracted tables, a sandboxed Python tool, and a VLM, and cites sources at page level.
- Implemented RAG from scratch (chunking, bge embeddings, FAISS, hand-written BM25, RRF fusion, cross-encoder reranking). On a 46-question benchmark whose answers are verified against the source PDFs, it raised Recall@5 from 81.5% (dense) to 88.0% and answer accuracy from 10.9% (closed-book) to 69.6%.
- Built an evaluation harness covering Recall@K, MRR, nDCG, answer accuracy, citation validity and precision, and agent tool-use metrics. Gold labels are auto-verified against the PDFs, and every ablation reports bootstrap 95% CIs.
- Trained the agent's tool-selection policy with policy-gradient RL, using a modular reward with potential-based shaping. It kept the same accuracy (67.4%) with 63% fewer tool calls (2.70 → 1.00) and 2× lower p50 latency.
- LoRA-fine-tuned Qwen2.5-0.5B on filtered agent trajectories, training 1.75% of parameters. On held-out steps, valid tool-call JSON rose from 71% to 100% and correct action choice from 45% to 71%.

**ML Engineer / ML Systems**
- Built an async FastAPI service with SSE progress streaming. Jobs run on an in-memory queue or on a Redis queue with idempotency keys and retries, and every request carries a trace id with Prometheus metrics.
- Load-tested the retrieval endpoint locally: 200 requests at concurrency 8, 13 req/s, 0 errors.
- Measured the no-batching baseline for inference: TTFT 229 ms and 19 tok/s at concurrency 1, with flat throughput at concurrency 4 on an HF server. This motivates vLLM's continuous batching; the vLLM deployment is configured but not yet measured.
- Diagnosed and fixed production-style failures: a PyTorch/FAISS OpenMP segfault, Metal thread-safety crashes (fixed with a process-wide inference lock), and a train/test leak in SFT data. The offline test suite covers unit, integration and e2e paths.

Do **not** claim that SFT improved end-to-end accuracy (it did not: 52.2% vs 58.7% for the base model). Do not claim any vLLM, Kubernetes or QLoRA measurements.
