# Roadmap

Milestones build on each other; every one is independently runnable and measurable, and earlier baselines are
kept so later versions can be compared in the same harness. Status legend matches
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

```
V0 handwritten RAG ──► V1 advanced RAG ──► V2 LangGraph agent ──► V4 multimodal
        │                    │                    │                     │
        └────────────────────┴──────► V3 evaluation harness ◄───────────┘
                                             │
                        trajectories ──► V5 SFT/LoRA/QLoRA ──► V6 agentic RL
                                             │
                               V7 serving (FastAPI, vLLM, Redis, Docker) ──► V8 distributed (K8s)
```

| Milestone | Scope | Depends on | Status |
|---|---|---|---|
| **V0 Handwritten RAG** | PDF/text/MD parsing, sentence chunking with overlap, bge embeddings, NumPy/FAISS cosine search, top-K, cited generation, CLI | – | Verified |
| **V1 Advanced RAG** | hand-written BM25, hybrid retrieval, RRF / weighted fusion, cross-encoder reranking, metadata filters, query rewrite, Qdrant adapter | V0 | Verified (Qdrant in embedded mode) |
| **V2 LangGraph agent** | planner / router / act / critic / writer, 7-action tool space, budgets, memory, fallbacks, streaming | V1 | Verified |
| **V3 Evaluation** | verified benchmark (46 Q), retrieval/generation/citation/agent/system metrics, experiment configs, ablation table with CIs | V0–V2 | Verified |
| **V4 Multimodal** | table extraction → SQLite + SQL tools, caption-anchored figure crops, VLM abstraction, unified evidence | V2, V3 | Verified (local VLM is weak; hosted/7B VLM configured) |
| **V5 Post-training** | trajectory schema/store/validation/filter, SFT data builder, TRL+PEFT LoRA trainer, QLoRA path, adapter eval/serving | V2, V3 | LoRA verified on MPS (small run); QLoRA requires CUDA |
| **V6 Agentic RL** | MDP env over production nodes, modular reward w/ potential shaping, REINFORCE actor-critic router policy, GRPO entry point for LLM router | V2, V3, V5 | Router RL verified; GRPO smoke-level only |
| **V7 Production serving** | FastAPI (sync/async/SSE), job queue (memory/Redis), worker, rate limiting, tracing, metrics, OpenAI-compatible client for vLLM, dev model server, Dockerfiles/Compose | V2 | API verified locally; vLLM & containers configured |
| **V8 Distributed** | K8s manifests: API HPA, KEDA-scaled workers, vLLM GPU deployment, Redis, Qdrant, ingress | V7 | Configured + statically validated, not deployed |

## Next steps (not implemented)
1. **Bigger, harder benchmark**: 200+ questions incl. multi-document synthesis, unanswerable questions (to measure
   abstention), chart-reading questions whose answers exist only in pixels.
   Planned v2 (keeping v1 frozen): ~160 PDF-verified questions over 12–16 table/figure-heavy arXiv papers, about 40%
   text / 25% table / 15% table + arithmetic / 10% figure / 10% cross-paper, so that multi-tool routing can pay off
   for the RL router (which on v1 collapses to a single search), plus a separately reported Qasper subset.
2. **Stronger judges**: NLI-based groundedness and an LLM-as-judge for answer correctness, calibrated against a
   human-labeled subset.
3. **Real vLLM measurements** on a GPU host: TTFT / tokens/s / throughput vs. concurrency, prefix caching on/off,
   multi-LoRA serving of the SFT adapter.
4. **GRPO at scale** on Qwen2.5-1.5B/7B with rollouts served by vLLM; compare base vs SFT vs SFT+GRPO agents.
5. **Layout-aware parsing** (borderless tables, multi-column reading order) and a 7B VLM for figures.
6. **OpenTelemetry exporter** for spans; Grafana dashboards for the Prometheus metrics.
7. **Incremental ingestion** (per-document upserts instead of full re-index) with Qdrant as the live store.
