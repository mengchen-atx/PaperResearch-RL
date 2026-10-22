# Atlas Architecture

This document explains **what each component does, why it exists, what the alternatives were, and what it costs**.
It is written to be studied: every section maps to a package in `src/atlas/` you can read alongside it.
For what is verified vs. only configured, see [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

```
                               ┌──────────────────────────────────────────────────────────┐
  User / UI ──HTTP/SSE──► API  │ FastAPI  (/ask /retrieve /research /research/{id}/events) │
                               └───────┬───────────────────────────────┬──────────────────┘
                                       │ sync                          │ async job (memory | Redis queue)
                                       ▼                               ▼
                              ┌──────────────────── LangGraph research agent ────────────────────┐
                              │  planner ─► router ─► act (tools) ─► critic ─┐                    │
                              │     ▲         │  ▲                           │ insufficient       │
                              │     └─replan──┘  └───────────────────────────┘                    │
                              │               └─answer / budget─► writer ─► cited report          │
                              └────┬───────────────┬──────────────┬──────────────┬───────────────┘
                                   │               │              │              │
                          search_documents   lookup_table     analyze_image   run_python / search_web
                                   │          query_sql           │              │
                    ┌──────────────▼──────┐  (SQLite tables)   VLM (SmolVLM /   sandboxed subprocess /
                    │ Hybrid retrieval     │                    Qwen-VL / Claude) Wikipedia, Tavily
                    │ BM25 + dense (FAISS/ │
                    │ Qdrant) → RRF → cross│
                    │ -encoder rerank      │
                    └──────────▲──────────┘
                               │
      PDFs/MD/TXT ──► ingestion: text layer · tables (find_tables→SQLite) · figures (caption-anchored crops)
                      └► chunking (sentence packing, page-bounded) └► embeddings (bge-small) + BM25 postings

  Every agent run ─► trajectory (JSONL) ─► quality filter ─► SFT (LoRA/QLoRA)  ─► adapter ─► hf_local / vLLM --enable-lora [configured, not validated]
                                        └► RL env (same nodes/tools) ─► router policy (REINFORCE) / LLM router (GRPO)
  Observability: trace ids on every request, spans for agent nodes / tools / retrieval / LLM calls, Prometheus metrics.
```

---

## 1. Knowledge layer: why RAG at all

**Problem.** An LLM's parametric memory is frozen at training time, is lossy for precise facts (numbers, table
cells), and gives no provenance. Our closed-book baseline (Qwen2.5-1.5B) answers **10.9%** of the benchmark
correctly; the same model with retrieved context answers **~60%** (see `docs/experiments/results.md`).

**Design.** Retrieval-augmented generation puts the relevant source passages in the prompt and asks the model to
answer only from them, citing `[n]` markers that map 1:1 to evidence objects. Citations become *measurable*
(`atlas/rag/citations.py` flags markers that point at nothing).

**Implemented by hand (no LangChain) on purpose** so every step is inspectable:
`ingestion/loaders.py → chunking/chunker.py → embeddings/embedder.py → retrieval/{vector_store,bm25,fusion,retriever}.py
→ rag/{prompts,pipeline,citations}.py`.

### 1.1 Parsing — native text, no OCR
arXiv-style PDFs are born-digital: PyMuPDF reads the text layer exactly and fast (4 papers, 67 pages ≈ 5 s
including table/figure extraction). OCR (Tesseract, Donut) is slower and introduces character errors; it is the
right tool only for scanned documents, which the loader detects and logs.

### 1.2 Chunking — sentence packing with overlap, never across pages
* **Why chunk:** embedding models have a 512-token window and one vector per paper would average every topic
  together; small chunks give precise retrieval *and* precise page citations.
* **How:** sentences are packed greedily up to `chunk_size` characters (1000 ≈ 250 tokens) with `chunk_overlap`
  (150) carried from the previous chunk so facts straddling a boundary survive intact in one chunk. Numbered
  section headings are detected and stored as metadata.
* **Alternatives:** fixed windows (simpler, cuts sentences), semantic chunking by embedding similarity (costly at
  ingest, marginal gains on papers), token-based sizing (model-specific). Characters keep the chunker model-agnostic.
* **Tradeoff:** chunks are page-bounded → a sentence split across a page break is split across two chunks.
  We accept that for exact page citations.

### 1.3 Embeddings and the vector index
* **Model:** `BAAI/bge-small-en-v1.5` (33M params, 384-d): strong retrieval quality per FLOP, runs on CPU/MPS.
  Queries get bge's instruction prefix (asymmetric retrieval).
* **Cosine similarity = dot product:** all vectors are L2-normalized, so `cos(q,d) = q·d`. The NumPy store is the
  literal textbook implementation (`scores = V @ q` + `argpartition` top-k); FAISS `IndexFlatIP` does the same math
  in SIMD C++; Qdrant is the production adapter (persistence, payload filters, replication).
* **Exact vs. approximate search:** at ~500 chunks exact search takes ~7 ms. ANN (HNSW/IVF) becomes worthwhile past
  ~10⁶ vectors; the `VectorStore` interface lets you swap it in without touching callers.
* **Embedding cache:** document embeddings are cached by `sha256(text)` per model, so re-ingesting or re-running
  ablations skips the encoder.

### 1.4 Why hybrid retrieval (BM25 + dense)
Dense retrievers capture paraphrase ("number of attention heads" ≈ "h = 8") but are weak on exact tokens that
dominate research questions: model names, dataset names, numbers ("WMT 2014", "BLEU", "d_k"). BM25 matches those
exactly but fails on paraphrase. Their failure modes are complementary. Measured on our benchmark:
Recall@5 dense 81.5 → BM25 85.9 → hybrid 87.0.

**Score fusion.** BM25 scores are unbounded, cosine is in [-1, 1], so raw scores can't be added.
* **RRF** (`Σ 1/(k+rank)`, k=60) uses ranks only → scale-free and tuning-free (default).
* **Weighted min-max** keeps score magnitudes but depends on the candidate distribution and needs a tuned weight.
Both are implemented and ablated.

### 1.5 Why a reranker
First-stage retrievers are *bi-encoders*: query and chunk are embedded independently, which is fast but never
lets the model compare them token-by-token. A *cross-encoder* (`ms-marco-MiniLM-L-6-v2`) reads `[query; chunk]`
jointly and is much more precise — but costs one forward pass per candidate. Two-stage design: retrieve
`candidate_k=30` cheaply, rerank those, keep `top_k=5`. Measured: Recall@5 87.0 → 88.0 at +150 ms/query on MPS.
**Tradeoff shown honestly in results:** better retrieval did not translate into better end-to-end accuracy for the
1.5B generator on this small benchmark (differences are within the bootstrap CI) — generator capacity, not
retrieval, is the bottleneck at this model size.

### 1.6 Metadata filtering
Pre-filtering (restrict the candidate set *before* scoring) by document, modality, or page range so Top-K is
computed within the allowed set; post-filtering a truncated list can return fewer than K results. FAISS uses an
`IDSelectorBatch`; Qdrant uses native payload filters.

---

## 2. Multimodal layer: route content to the cheapest correct extractor

| Content | Extractor | Why |
|---|---|---|
| Running text | PDF text layer | exact, free |
| Ruled tables | `page.find_tables()` → markdown chunk **and** SQLite rows | LLMs read tables in prompts but can't reliably compute over them; SQL gives exact aggregation |
| Figures / charts | caption-anchored crop rendered to PNG; caption indexed as a FIGURE chunk | captions are text (no OCR); pixels are only sent to a VLM when a question needs visual reasoning |

* **Figure localization:** find "Figure N" caption blocks, union the vector drawings/images directly above them,
  render that region (`multimodal/pdf_extract.py`). This works for vector charts (the scaling-laws paper), which
  image-extraction approaches miss entirely.
* **Table detection is imperfect** on real papers (borderless tables are missed; dense attention heatmaps look
  like tables). Detected tables are filtered by shape/density heuristics; missed tables remain available as text.
* **VLM abstraction** (`multimodal/vlm.py`): `hf_local` (SmolVLM-256M, offline), `openai_compatible` (vLLM-served
  Qwen2-VL etc.), `anthropic`. The local 256M model runs (≈4 s/figure on MPS) but reads charts poorly — it is
  there to prove the pipeline offline; production should use a 7B-class VLM or a hosted one.
* **Unified evidence:** every tool returns `Evidence(evidence_id, modality, content, source_tool, filename, page,
  score, metadata{table_id|figure_id})`, so the writer can cite a table cell, a figure reading, a SQL result or a
  Python computation the same way it cites a paragraph.

---

## 3. The agent: why LangGraph, and why tools are separate

**Problem.** Single-shot RAG retrieves once for the literal question. Research questions often need several
steps: decompose ("compare X in paper A with Y in paper B"), fetch a table, compute a difference, notice missing
evidence and search again.

**Graph** (`agents/graph.py`): `planner → router → act → critic → router … → writer`, plus `router → planner`
for re-planning.

* **Planner** decomposes into ≤3 sub-questions (LLM, JSON-validated, heuristic fallback).
* **Router** chooses one action from a fixed action space: `search_documents, lookup_table, analyze_visual,
  run_python, search_web, replan, answer`. The policy is pluggable: **heuristic** (transparent rules),
  **llm** (JSON actions, heuristic fallback on invalid output, fallbacks are counted as a metric), **learned**
  (RL-trained MLP). One action space for all three makes them directly comparable.
* **Act** executes the tool via the `ToolExecutor`; the *Python Analyst* sub-step lets the LLM write code over
  numeric evidence which the sandbox executes.
* **Critic** judges sufficiency per sub-question (term coverage of evidence, or an LLM judge) and marks
  sub-questions resolved or exhausted.
* **Writer** selects a diverse evidence set (computations/tables/figures first, then text round-robin across
  sub-questions) and writes a cited answer; the report lists every source with a `cited` flag.

**Why LangGraph over a hand-written loop or LangChain's AgentExecutor:** typed state with reducers, cycles with
conditional edges as first-class, per-node streaming (powers the SSE progress API), and checkpoint/interrupt
support for future human-in-the-loop — while node logic stays plain Python (`agents/nodes.py`) and unit-testable.
LangChain abstractions are not used *inside* the core: they would hide exactly the fundamentals this project exists to show. For interoperability, `atlas/integrations/langchain.py` exposes the hand-written retriever as a LangChain `BaseRetriever` and every tool as a `StructuredTool` (same pydantic schemas, executed through Atlas's `ToolExecutor` so timeouts, metrics and tracing still apply). Any LangChain/LangGraph application can use Atlas without knowing its internals.

**Loop safety:** `max_iterations`, `max_tool_calls`, a wall-clock deadline, and LangGraph's recursion limit as a
backstop. A failing tool or policy is recorded, never fatal.

**Why tools are separate units** (`tools/`): each has a pydantic argument schema (validation of LLM-produced
arguments), a structured result, and uniform instrumentation (latency, errors, Prometheus counters, trace spans).
The executor enforces timeouts and converts crashes into recorded errors. The resulting `ToolCallRecord`s are the
raw material for agent metrics and for post-training data.

**Memory.** Working memory = the agent state: call signatures already executed (identical calls are reformulated
once, then skipped and counted as waste), evidence deduplicated by id and content, unresolved sub-questions.
Persistent memory = `memory/history.py` (SQLite research sessions, BM25 lookup of related past queries).

---

## 4. Evaluation as a first-class component

`evaluation/` defines a verified benchmark format and pure-function metrics:
* **Retrieval:** Recall@K, Precision@K, MRR, nDCG@K with page-level relevance (each gold evidence unit credited once).
* **Answers:** numeric match within tolerance for numeric questions, boundary-aware containment of gold/aliases
  otherwise (so "8" does not match "80"), token F1.
* **Groundedness:** fraction of answer sentences whose content words are mostly present in the evidence — a
  *lexical proxy*, reported as such (an NLI or LLM-judge would be stronger and is a roadmap item).
* **Citations:** validity (markers that point at real evidence), precision (cited evidence on a gold page),
  coverage (sentences with a citation). Undefined values are kept undefined instead of averaged as 0.
* **Agent:** success, completion, tool calls, failed/unproductive/duplicate calls, replans, policy fallbacks,
  *required-tool recall* (did a table question use a table tool?).
* **System:** p50/p95 latency, tokens, estimated cost, throughput; API load test; inference TTFT/tok-s.

Gold labels are **verified against the PDFs** by `scripts/build_benchmark.py` (it refuses to write the benchmark if
any quote is missing from its labeled page). Results tables are generated from per-item records with bootstrap
95% CIs — with 46 questions, one question is 2.2 points.

---

## 5. Post-training: why SFT on trajectories

Small open models (0.5B–1.5B) often fail the agent protocol: invalid JSON actions, wrong tools. Rather than
prompting harder, we **post-train the model on the agent's own successful behavior** (rejection sampling /
expert iteration): trajectories that answered correctly *with valid citations* pass `quality_filter`, and each
yields router examples (state at step t → JSON action, with the exact evidence prefix reconstructed) and a writer
example (grounded prompt → cited answer). Loss is on completion tokens only; train/val split by trajectory (no
leakage).

**LoRA** trains a low-rank update `W + (α/r)·BA` on attention/MLP projections — ~1–2% of parameters, trainable on
a laptop, adapters are MBs and hot-swappable in vLLM (`--enable-lora`). **QLoRA** (configured, not validated — no CUDA GPU) loads the frozen base in 4-bit
NF4 (bitsandbytes) to fit 7B-class models on one 24 GB GPU; it requires CUDA and fails fast elsewhere.
We never train a foundation model from scratch — we post-train an existing open model.

---

## 6. Agentic RL: why, and what is actually optimized

SFT imitates past behavior; it cannot discover *better* tool-use strategies (fewer calls, knowing when to stop).
RL optimizes a reward that encodes what we care about.

**MDP** (`rl/env.py`): state = research state; observation = fixed feature vector (query cues, evidence
modalities, coverage, budget use, action history) + action mask; actions = the agent's action space; transitions =
**the production agent's own node functions and tools**, so a trained policy transfers without adaptation.

**Reward** (`rl/reward.py`, all weights configurable):
`R = α·task_success + β·citation_quality + γ·evidence_quality − λ·Σ tool_cost − μ·Σ steps − ν·invalid`,
where evidence quality uses **potential-based shaping** (Φ = recall of gold evidence pages), which densifies the
signal without changing the optimal policy.

**Algorithms.**
* Router MLP: REINFORCE with a learned value baseline + entropy bonus (`rl/train.py`). Tool results are memoized
  (deterministic tools), making ~1000 episodes feasible on a laptop. The terminal writer inside RL rollouts is the
  extractive answerer (cheap); this is a deliberate approximation, documented in `docs/design/agentic_rl.md`.
* LLM router: GRPO via TRL (`rl/grpo.py`) with Monte-Carlo rollouts in the same environment as the reward — a
  one-step policy-improvement formulation. Needs a GPU for real runs; smoke-tested on CPU/MPS.

---

## 7. Serving: why vLLM, async jobs, and caching

> vLLM status: configured, not validated (no GPU available). The serving path was run end to end against the HF dev server (`atlas serve-model`).

* **vLLM** for production inference: continuous batching (new requests join in-flight batches instead of waiting),
  PagedAttention (KV cache in fixed-size blocks → little fragmentation → more concurrent sequences), prefix caching
  (agent prompts share long system/evidence prefixes), multi-LoRA serving for SFT adapters, tensor parallelism for
  large models. The app talks to it through the `openai_compatible` provider, the same client used for any
  OpenAI-style endpoint. On machines without NVIDIA GPUs, `atlas serve-model` exposes the identical API over HF
  transformers so the full path (client, streaming, benchmark) runs end-to-end.
* **Async research jobs:** a research run takes tens of seconds; holding HTTP connections that long breaks behind
  load balancers. `POST /research` returns 202 + job id; progress streams over SSE. Memory backend (in-process
  workers) for single-node; Redis backend (shared job state + work queue + `atlas worker` processes) for
  distributed mode. At-least-once delivery with idempotent job handling, bounded retries, `Idempotency-Key`
  support.
* **Caching only where safe:** retrieval results (index immutable per build), LLM responses at temperature 0,
  document embeddings. Not cached: sampled generations, web search (results drift).
* **Robustness:** request validation (pydantic, `extra="forbid"`), timeouts around blocking work, typed errors →
  JSON responses with request ids, rate limiting, upload validation (type, size, path traversal), read-only SQL,
  sandboxed Python disabled in production by config validation.

## 8. Distributed architecture: why each piece exists

> Status: configured, not validated (no GPU/cluster available). Everything below is design + manifests.

| Component | Problem it solves | When to add it |
|---|---|---|
| API replicas + HPA (CPU) | request fan-in, embedding/rerank CPU load | >1 node or HA needed |
| Redis job queue + workers (KEDA on queue depth) | long-running agent jobs decoupled from HTTP; scale workers on backlog, not CPU (workers mostly wait on inference) | concurrent research jobs |
| vLLM Deployment (1 GPU/replica) | GPU-efficient batched inference, shared by all workers | any real traffic |
| Qdrant StatefulSet | vector search shared across replicas, persistence, payload filters | index outgrows per-pod FAISS or needs live updates |
| Shared RWX volume / object storage | chunks, BM25 postings, figures, table DB visible to all pods | multi-pod |
| Ingress (rate limit, long SSE timeouts) | gateway concerns out of the app | public exposure |

Single-node mode (`atlas serve`, memory queue, FAISS, in-process workers) remains the default and is what the
benchmarks ran on. The Kubernetes manifests are **configured, not validated** (no cluster was available; only statically
checked in tests); see IMPLEMENTATION_STATUS.md.

## 9. Observability
Every request gets an `X-Request-ID` which is the trace id; spans are recorded for the request, each agent node,
each tool call, retrieval and each LLM/VLM call (latency, tokens, status) and exported as JSONL per trace
(`GET /traces/{id}`). Logs are structured (JSON in containers) and carry the trace id. Prometheus metrics cover
requests, tool calls, LLM tokens/latency, retrieval latency, cache hits, jobs. Vendor-neutral by design; an OTel
exporter can consume `SpanRecord`s.
