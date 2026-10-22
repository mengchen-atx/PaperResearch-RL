# Implementation status

Environment for every verification below: Apple M4 (10 cores, 16 GB, MPS), macOS, Python 3.12, **no NVIDIA
GPU, no Docker daemon, no Kubernetes cluster, no hosted-LLM API keys.** Local models: Qwen2.5-1.5B-Instruct
(generator), Qwen2.5-0.5B-Instruct (SFT/GRPO), bge-small-en-v1.5, ms-marco-MiniLM-L-6-v2, SmolVLM-256M-Instruct.

Legend
- **VERIFIED**: implemented and executed end to end here, with tests and/or benchmark output in the repo.
- **IMPLEMENTED, NOT FULLY EXECUTED**: code complete and exercised by tests or a smoke run, but the full-scale run was not performed.
- **CONFIGURED, REQUIRES EXTERNAL RESOURCE** (not validated): config and code exist; running it needs hardware or services not available here.
- **NOT IMPLEMENTED**

| Component | Status | Evidence / notes |
|---|---|---|
| RAG (V0: parse → chunk → embed → FAISS/NumPy cosine → top-K → cited answer) | **VERIFIED** | `atlas ingest` + `atlas ask` run end to end; `tests/integration/test_pipeline.py`; `benchmarks/results/vanilla_rag` |
| Hybrid RAG (BM25 + dense, RRF / weighted fusion, metadata filters) | **VERIFIED** | `benchmarks/results/retrieval_*`, `hybrid_rag`; unit tests for BM25, fusion, filters |
| Reranker (cross-encoder) | **VERIFIED** | `retrieval_*_rerank`, `hybrid_rerank_rag` |
| Qdrant adapter | **VERIFIED (embedded mode)** / CONFIGURED (server) | same results as FAISS/NumPy in `tests/unit/test_vector_stores.py` (qdrant-client local mode); server mode configured in compose/K8s, not run |
| Multimodal: tables | **VERIFIED** | `find_tables` → markdown chunks + SQLite; `lookup_table` / `query_sql` tools used by the agent; detection misses borderless tables (documented) |
| Multimodal: figures + VLM | **VERIFIED (local 256M VLM)** | caption-anchored crops for vector charts; `analyze_image` runs SmolVLM-256M on MPS (~4 s). Chart-reading quality at this size is poor. OpenAI-compatible and Anthropic VLM adapters are unit-tested with mock transports only (no keys) |
| LangGraph agent (planner / router / act / critic / writer, budgets, memory, fallbacks) | **VERIFIED** | `tests/integration/test_agent.py`; `benchmarks/results/agent_*`; SSE streaming through the API |
| Tools: search_documents, read_document, lookup_table, query_sql, run_python (sandbox), analyze_image | **VERIFIED** | unit tests incl. sandbox escape attempts, SQL injection guards, timeouts, crash isolation |
| Tool: search_web | **IMPLEMENTED, NOT FULLY EXECUTED** | Wikipedia (no key) and Tavily backends; disabled by default so benchmarks stay reproducible; not exercised against the live web in tests |
| LangChain adapters (BaseRetriever + StructuredTool wrappers) | **VERIFIED** | `tests/integration/test_langchain_adapter.py`; smoke-tested on the real index |
| Memory (working state dedupe + persistent SQLite history) | **VERIFIED** | unit tests; `/sessions` endpoint |
| Evaluation framework (retrieval, answer, groundedness proxy, citations, agent, system metrics) | **VERIFIED** | 46-question benchmark verified against PDFs; `docs/experiments/results.md` generated from records |
| Trajectory collection, validation, filtering | **VERIFIED** | `data/trajectories/*.jsonl` from agent runs; `atlas trajectories validate`; unit tests |
| SFT (TRL + PEFT) | **VERIFIED (small scale)** | Qwen2.5-0.5B-Instruct, 29 filtered trajectories (held-out questions excluded) → 123 train / 31 val router examples, 2 epochs, 11 min on MPS; `data/processed/sft/{train_metrics,eval}.json`. Router-only: the trajectories were collected before the metadata-merge fix, so writer examples were unavailable (fixed + regression-tested; re-collect to include them). Imitation improved (valid JSON 71%→100%, action acc. 45%→71%), but the SFT agent scored 52.2% vs 58.7% for the un-adapted 0.5B control (`agent_base05`) |
| LoRA | **VERIFIED (small scale)** | adapter saved to `data/processed/sft/adapter` (not committed); `llm.adapter_path` loads it for inference |
| QLoRA | **CONFIGURED, REQUIRES EXTERNAL RESOURCE** | `configs/sft/qlora.yaml`, NF4 path in `training/sft.py`; bitsandbytes 4-bit needs CUDA, and non-CUDA devices fail fast by design |
| Agentic RL (env, modular reward, REINFORCE actor-critic router) | **VERIFIED (small scale)** | `data/processed/rl/results.json`: 960 episodes, evaluated on 13 held-out questions vs heuristic and random; extractive writer inside rollouts (documented approximation) |
| Agentic RL: GRPO for the LLM router | **IMPLEMENTED, NOT FULLY EXECUTED** | `rl/grpo.py`, `scripts/train_grpo.py --smoke` executed: 2 GRPO steps on Qwen2.5-0.5B + LoRA (MPS), rewards from real env rollouts (mean reward 0.20 → -0.00; 2 steps show the pipeline works, not learning). A meaningful run needs a CUDA GPU |
| FastAPI service (sync, async jobs, SSE, uploads, traces, metrics, rate limit) | **VERIFIED** | `tests/integration/test_api.py`; local load test in `benchmarks/results/system` |
| vLLM | **CONFIGURED, REQUIRES EXTERNAL RESOURCE** | OpenAI-compatible client verified against mock transport and the bundled HF dev server (`atlas serve-model`); vLLM compose/K8s configs written; **no vLLM numbers measured** (needs NVIDIA GPU) |
| Inference benchmark (TTFT, tok/s, throughput) | **VERIFIED against HF dev server** | `benchmarks/results/inference/*.json`. These are not vLLM numbers |
| Redis (cache, job store, queue, worker) | **IMPLEMENTED, NOT FULLY EXECUTED** | logic verified against `fakeredis` (Redis protocol emulator) incl. idempotency and redelivery; no real Redis server was available |
| Docker / Compose | **CONFIGURED, REQUIRES EXTERNAL RESOURCE** | Dockerfile + compose written; compose YAML parsed in tests; **images never built** (no Docker daemon) |
| Kubernetes manifests (API HPA, KEDA workers, vLLM GPU, Redis, Qdrant, ingress) | **CONFIGURED, REQUIRES EXTERNAL RESOURCE** | parsed and statically checked (resources, probes, production ConfigMap validates as an AtlasConfig) in `tests/unit/test_deploy_config.py`; never applied to a cluster |
| Distributed deployment | **CONFIGURED, REQUIRES EXTERNAL RESOURCE** | architecture + manifests only; single-node mode is what ran |
| Observability (trace ids, spans, JSON logs, Prometheus) | **VERIFIED** | `/traces/{id}` returns request → node → tool → LLM spans (API test); `/metrics` |
| OpenTelemetry / Grafana export | **NOT IMPLEMENTED** | roadmap |
| Web UI | **IMPLEMENTED, NOT FULLY EXECUTED** | `src/atlas/serving/static/index.html` served at `/` (the route is tested); not click-tested in a browser in this session |
| LLM-as-judge / NLI groundedness | **NOT IMPLEMENTED** | groundedness is a lexical proxy, labeled as such |

## Known limitations
- Small benchmark (46 questions, 4 papers, one domain). Many ablation differences fall inside the 95% CIs.
- Generator is a 1.5B local model. Several errors come from the generator misreading numbers that are correct in the retrieved context.
- LLM routers (1.5B and 0.5B) rarely choose to stop and mostly end at the tool budget.
- RL and SFT runs are pipeline-scale (tens of questions / trajectories). They show direction, not general gains.
- Table detection only handles ruled tables. SmolVLM-256M chart reading is unreliable.
- Latencies were measured on a laptop GPU with other jobs sometimes running concurrently. Treat them as indicative only.
