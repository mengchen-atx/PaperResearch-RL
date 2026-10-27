# Atlas

Research questions over a paper collection: retrieve, use tools, cite the page, and keep the run as training data.

[![CI](https://github.com/mengchen-atx/PaperResearch-RL/actions/workflows/ci.yml/badge.svg)](https://github.com/mengchen-atx/PaperResearch-RL/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-A31F34)](LICENSE)

![Atlas: PDFs to retrieval, a LangGraph loop, a cited answer, and SFT / RL data](docs/assets/hero.png)

## Quickstart

CPU only. No model downloads. About two minutes after the install.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make quickstart PY=python
```

This path uses a hashing embedder and an extractive answerer, and it writes `data/processed/quickstart_index`. It checks that ingest, retrieval, and the agent loop run. The numbers below used bge-small, the cross-encoder, and Qwen2.5-1.5B. Full setup is [further down](#full-setup).

## Headline results

46 questions, gold answers and page quotes checked against the PDFs. Same benchmark for every row. The 0.5B SFT run is a different generator, so it is not on this chart. Intervals are wide: one question is 2.2 points.

<!-- BEGIN GENERATED RESULTS -->

![Accuracy versus p50 latency. Squares are agents.](docs/assets/accuracy_latency.png)

![Recall@5 and p50 latency for each retrieval variant.](docs/assets/recall_at_5.png)

Accuracy is the mean of per-question `correct` in each `records.jsonl`, with a 95% bootstrap CI (seed 0, 5000 resamples; same procedure as `scripts/make_results_table.py`). p50 latency is `latency_p50_ms` in `summary.json`. Device `mps`. Summary dates 2026-09-29 to 2026-10-21. Each file's platform and timestamp is in `docs/assets/figure_data.json`.

| System | Answer accuracy (95% CI) | Tool calls | p50 latency |
|---|---|---|---|
| Closed-book LLM (no retrieval) | 10.9% (2.2–19.6) | – | 1.4 s |
| No-LLM floor: hybrid+rerank retrieval, extractive answerer | 41.3% (28.3–54.3) | – | 167 ms |
| Vanilla RAG: dense top-5, no reranker | 63.0% (50.0–76.1) | – | 9.5 s |
| Hybrid RAG: BM25 + dense RRF top-5, no reranker | 58.7% (43.5–71.7) | – | 5.4 s |
| Hybrid RAG + cross-encoder reranker, top-5 | 69.6% (56.5–82.6) | – | 5.3 s |
| Agent ablation: no VLM, no tables/SQL, no Python (text retrieval only) | 67.4% (54.3–80.4) | 1.93 | 11.5 s |
| LangGraph agent, rule-based router, LLM planner/writer, all tools incl. VLM | 67.4% (52.2–80.4) | 2.70 | 19.6 s |
| LangGraph agent, LLM router (JSON actions) with heuristic fallback | 69.6% (56.5–82.6) | 5.52 | 31.2 s |
| Agent with RL-trained router policy (REINFORCE, see atlas.rl) | 67.4% (54.3–80.4) | 1.00 | 9.2 s |
| Fixed baseline: one search_documents call, then writer (no RL) | 67.4% (54.3–80.4) | 1.00 | 8.6 s |

<!-- END GENERATED RESULTS -->

Retrieval is the large gap. Closed-book accuracy is 10.9%. With retrieval, the same 1.5B generator lands between 58.7% and 69.6%. Agents match that band. They do not beat the best single-shot RAG. The RL router reaches 67.4% with 1.00 tool call, the same accuracy as the fixed one-search baseline (8.6 s vs 9.2 s p50). `required_tool_recall` in `benchmarks/results/agent_learned/summary.json` is 0.0, so that policy did not use the table, SQL, or vision tools. Recall@5 goes from 81.5 (dense) to 88.0 once a cross-encoder reranks hybrid or dense candidates. Hybrid without rerank is 87.0 at about 7 ms p50; hybrid plus rerank is 88.0 at 232 ms p50.

The candid reading of the RL result, the SFT result, and what was not run: [below](#what-these-numbers-say). Full tables: [`docs/experiments/results.md`](docs/experiments/results.md). Regenerate the charts with `python scripts/make_readme_figures.py`.

## Architecture

```mermaid
flowchart LR
  subgraph ingest ["Ingestion and index"]
    docs["PDF / MD / TXT"] --> chunks["page-bounded chunks, tables, figure crops"]
    chunks --> bm25["BM25"]
    chunks --> dense["bge-small, FAISS or Qdrant"]
    bm25 --> fusion["RRF or weighted, then cross-encoder"]
    dense --> fusion
  end

  subgraph agent ["LangGraph"]
    question["question"] --> planner["planner"]
    planner --> router["router"]
    router -->|"tool action"| act["act"]
    act --> critic["critic"]
    critic --> router
    router -->|"replan"| planner
    router -->|"answer or budget"| writer["writer"]
    writer --> report["cited report"]
    act --> tools["search_documents, lookup_table, analyze_visual, run_python, search_web"]
  end

  fusion --> act
  report --> traj["trajectory JSONL"]
  traj --> filt["quality filter"]
  filt --> sft["SFT LoRA, loaded into the local LLM"]
  filt --> rl["REINFORCE on the same nodes"]
  rl -.-> router
```

![Retrieval, the LangGraph loop, and the trajectory to SFT / RL](docs/assets/architecture.png)

The edges match `src/atlas/agents/graph.py`: `START → planner → router → act → critic → router`, `replan → planner`, and `answer` or the budget → `writer → END`. Router actions are the `Action` enum in `src/atlas/agents/actions.py`. Policies are heuristic, LLM JSON with a heuristic fallback, a learned MLP, and a fixed single-search baseline. `search_documents` is BM25 plus dense, then RRF or weighted fusion, then an optional cross-encoder (`src/atlas/retrieval/retriever.py`).

`query_sql` and `read_document` are registered on the tool executor. The router does not select them. QLoRA and vLLM serving are configured, not run. GRPO completed a 2-step smoke test, not a full training run. Status of every piece: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

Why each piece exists, with the alternatives: **[ARCHITECTURE.md](ARCHITECTURE.md)**.
Design notes: [RAG](docs/design/rag.md) · [post-training](docs/design/post_training.md) · [agentic RL](docs/design/agentic_rl.md) · [serving](docs/design/serving.md) · [evaluation](docs/design/evaluation.md).
Milestones: [ROADMAP.md](ROADMAP.md).

## Features

| Area | What is implemented | Status |
|---|---|---|
| RAG from scratch | PyMuPDF parsing, sentence-packing chunker, bge-small embeddings, NumPy and FAISS cosine search, cited generation | Verified |
| Advanced retrieval | Okapi BM25 (hand-written), hybrid RRF / weighted fusion, cross-encoder reranking, metadata pre-filtering, Qdrant adapter | Verified |
| Multimodal | ruled-table extraction to SQL tools; vector-chart figure crops; VLM abstraction (local SmolVLM, OpenAI-compatible, Anthropic) | Verified (weak local VLM) |
| LangChain interop | `AtlasRetriever` (`BaseRetriever`) and `StructuredTool` wrappers for every Atlas tool | Verified (tests + real index) |
| Agent | LangGraph state machine, 7-action tool space, interchangeable policies, critic loop, budgets, dedupe, fallbacks | Verified |
| Evaluation | PDF-verified benchmark; retrieval, answer, groundedness, citation, agent, and system metrics; bootstrap CIs | Verified |
| Post-training | trajectory schema, filtering, trajectory to SFT data, TRL + PEFT LoRA, QLoRA path, base-vs-adapter eval | LoRA verified at small scale; QLoRA configured, not validated |
| Agentic RL | MDP over the production nodes, shaped reward, actor-critic REINFORCE, GRPO entry point | Router RL verified at small scale; GRPO smoke only |
| Serving | FastAPI sync / async / SSE, memory or Redis queue, worker, uploads, traces, metrics, dev model server | Verified locally; vLLM configured, not validated |
| Deployment | Dockerfile, Compose, Kubernetes manifests (HPA, KEDA, GPU vLLM, StatefulSets, ingress) | Configured, not validated; manifests statically checked |

## A recorded run

Unedited figure scenario from [`docs/experiments/demo_output.md`](docs/experiments/demo_output.md), rendered by `scripts/make_readme_figures.py`. The image caption is checked against benchmark item `fig-01`. The same file includes runs that get the answer wrong. That is the file the limitations section points at.

![Recorded heuristic-policy run: question, tool trace, answer, page citations](docs/assets/demo_run.png)

The figure above is a still of that recorded run. To replace it with a terminal GIF from your machine:

```bash
make quickstart PY=python          # CPU, no model downloads
python scripts/demo.py             # full models; rewrites docs/experiments/demo_output.md
```

## Why this project exists

LLMs answer research questions fluently and unreliably. They misremember numbers, cannot show sources, and fail on facts that live in tables or charts. Atlas is a working, measured take on five engineering questions:

1. How much does retrieval help, and which pieces (BM25, dense, fusion, reranking) matter?
2. When does a multi-step agent beat single-shot RAG, and what does it cost in tool calls and latency?
3. Can a small open model learn the agent protocol from its own successful trajectories (SFT)?
4. Can RL make tool use more efficient without losing accuracy?
5. What does it take to serve this reliably: async jobs, streaming, caching, vLLM, horizontal scaling?

## Full setup

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), about 6 GB of disk for models. A GPU is optional. CUDA or Apple MPS is used when present.

```bash
make setup      # uv venv + pip install -e ".[train,vision,dev]"
make data       # 4 public arXiv papers (v1) and a verified benchmark
make ingest     # atlas ingest data/raw → text, table, and figure chunks
make test       # offline tests, no model downloads
```

The first full run downloads Qwen2.5-1.5B-Instruct, bge-small-en-v1.5, ms-marco-MiniLM-L-6-v2, and SmolVLM-256M-Instruct.

### Ask (single-shot RAG)

```bash
python -m atlas ask "What is self-attention?"
```

Real output, abridged. Answer, then evidence with document, page, section, retriever, and score:

```
Answer (Qwen/Qwen2.5-1.5B-Instruct):
Self-attention, also known as intra-attention, is an attention mechanism ... relating different positions of a
single sequence ... [1] ...

Retrieved evidence:
  [1] arxiv_1706.03762.pdf, p.2 score=9.5273 via=hybrid-rrf+cross_encoder section='2 Background'
  [2] arxiv_1706.03762.pdf, p.6 score=4.9797 via=hybrid-rrf+cross_encoder section='4 Why Self-Attention'
  ...
latency: retrieval 596 ms, generation 9792 ms; tokens: 825 in / 167 out
```

### Research (multi-step agent)

```bash
python -m atlas research "By how many BLEU points does Transformer (big) outperform the ConvS2S Ensemble on EN-DE?"
python -m atlas research "..." --policy llm       # LLM router
python -m atlas research "..." --policy learned   # RL-trained router
```

It prints the plan, every router decision and tool call, critic verdicts, and a report with cited sources. The trajectory is appended to `data/trajectories/trajectories.jsonl`.

### Serve

```bash
python -m atlas serve --port 8000      # UI at http://localhost:8000, OpenAPI at /docs
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question":"What is BM25?"}'
curl -s localhost:8000/research -H 'content-type: application/json' -d '{"query":"What does Figure 1 of the RAG paper show?"}'
# → 202 {"job_id": ..., "events_url": "/research/<id>/events"}
curl -N localhost:8000/research/<id>/events
```

### Another model provider

```bash
ATLAS__LLM__PROVIDER=openai_compatible ATLAS__LLM__BASE_URL=http://gpu-host:8000/v1 ATLAS__LLM__MODEL=Qwen/Qwen2.5-7B-Instruct python -m atlas ask "..."   # vLLM — configured, not validated
ATLAS__LLM__PROVIDER=anthropic ATLAS__LLM__MODEL=claude-haiku-4-5 ANTHROPIC_API_KEY=... python -m atlas ask "..."
ATLAS__LLM__PROVIDER=extractive python -m atlas ask "..."   # no LLM, fully offline
```

## Honesty contract

> Every number in this README is generated by a command in this repo and stored under `benchmarks/results/` or `data/processed/`. What ran versus what is only configured is listed in [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md). No vLLM, Kubernetes, or QLoRA numbers are claimed, because no NVIDIA GPU or cluster was available.

The SFT and RL training JSON under `data/processed/` is gitignored. The committed summary of those runs is [`docs/experiments/results.md`](docs/experiments/results.md), produced by `scripts/make_results_table.py`. The charts in this file are recomputed from `benchmarks/results/` only.

## What these numbers say

- Retrieval is the big win (10.9% → 63–70%).
- Agents match, but do not beat, the best single-shot RAG on this benchmark.
- The RL router does not beat a fixed "search once, then answer" pipeline.
- SFT taught the agent protocol but did not improve answers.

**About the RL result.** The REINFORCE router reaches the heuristic router's accuracy (67.4% vs 67.4% on 46 questions) with 1.00 vs 2.70 tool calls and 9.2 s vs 19.6 s p50 latency. It does this by collapsing to a single `search_documents` call on every question: `required_tool_recall` is 0.0 in `benchmarks/results/agent_learned/summary.json`, so it never uses the table, SQL, or vision tools. On this benchmark one good hybrid + rerank search is usually enough, and the tool-cost penalty (`lambda_tool_cost` in `configs/rl/reinforce.yaml`) makes extra calls unprofitable. The fixed `agent_single_search` baseline (one search, then the writer, no learning) is correct on exactly the same 46 questions as the learned router (paired accuracy difference 0.0, bootstrap 95% CI [0.0, 0.0]; 42 of 46 answers are textually identical), at 8.6 s p50. So the learned policy is equivalent to that fixed pipeline here. Testing whether RL can learn *when* extra tools pay off needs a benchmark with many more table, arithmetic, and figure questions ([ROADMAP.md](ROADMAP.md)).

**About SFT.** LoRA on Qwen2.5-0.5B, 31 held-out router steps: valid JSON 71.0% → 100%, correct action 45.2% → 71.0%. End-to-end, the adapter did not help: 52.2% versus 58.7% for the same 0.5B model without it (`agent_sft` vs `agent_base05`). Those runs are in [`docs/experiments/results.md`](docs/experiments/results.md).

## Design summary

**RAG.** Chunks are sentence-packed to about 1000 characters with 150 characters of overlap and never cross pages, so each chunk has one exact page citation. Retrieval is hybrid: BM25 catches exact tokens (model names, numbers), dense retrieval catches paraphrase. RRF fuses the two by rank, which avoids score-scale mismatch. A cross-encoder reranks the top 30. The prompt numbers evidence `[n]`, and the papers' own bibliography brackets are rewritten to `(ref n)` so the model cannot confuse them with our citations. Invalid markers are detected.

**Agent.** A LangGraph cycle of planner → router → act → critic, which then loops or hands off to the writer. The router picks from a fixed action space, so heuristic, LLM, and RL policies are interchangeable and comparable. Budgets (iterations, tool calls, wall clock), duplicate-call memory, and fallbacks prevent runaway loops. Failures are recorded, not fatal.

**Multimodal.** Content goes to the cheapest correct extractor. Text comes from the text layer. Ruled tables go to SQLite, so arithmetic is exact. Figure crops are located via their captions (this works for vector charts), and a VLM is called only when a question needs pixels.

**Post-training.** The agent's own successful, citation-valid trajectories become SFT data: router steps are rebuilt with the exact evidence prefix, plus writer examples. Training uses LoRA on a small Qwen model, with a QLoRA path for 7B-class models on CUDA (configured, not validated). Evaluation compares action accuracy on held-out trajectories.

**Agentic RL.** The MDP's transitions are the production node functions, so a trained policy runs unchanged in the agent. The reward is task success + citation precision + potential-shaped evidence recall − tool cost − steps. The router is trained with REINFORCE plus a learned baseline. A GRPO path exists for LLM routers.

**Serving.** Research runs are async jobs (202 + SSE) backed by an in-process queue or by Redis plus workers (idempotency keys, at-least-once delivery, bounded retries, graceful shutdown). Caching covers retrieval, temperature-0 LLM calls, and embeddings. vLLM would sit behind the same OpenAI-compatible client (configured, not validated; locally an HF dev server stands in). Every request carries a trace id, and spans cover nodes, tools, retrieval, and LLM calls.

**Distributed.** API replicas scale on CPU (HPA). Workers scale on queue depth (KEDA), because they mostly wait on inference. vLLM runs one GPU per replica, with prefix caching and multi-LoRA. Qdrant serves shared vector search. The ingress handles rate limiting and SSE timeouts. All of this is configured, not validated. Manifests are only statically checked.

## Evaluation methodology

* **Benchmark**: `data/benchmarks/atlas_papers_v1.jsonl`, 46 questions over *Attention Is All You Need*, BERT, RAG (Lewis et al.), and *Scaling Laws for Neural Language Models*. By type: 28 text, 8 table, 4 compute, 4 figure, 2 multi-hop. Each question has a gold answer and aliases, and numeric questions have a tolerance. Gold evidence is recorded as (file, pages, quote), and `scripts/build_benchmark.py` checks every quote against the actual PDF page before writing the file.
* **Retrieval**: Recall@K, Precision@K, MRR, and nDCG@K, with page-level relevance where each gold unit is credited once.
* **Answers**: numeric match within tolerance, or boundary-aware containment of the gold answer or an alias, plus token F1.
* **Groundedness**: share of answer sentences whose content words appear in the evidence. This is a lexical proxy, not an NLI judgment.
* **Citations**: validity (markers point to real evidence), precision (cited evidence is on a gold page), and coverage.
* **Agent**: success, completion, tool calls, failed / unproductive / duplicate calls, replans, policy fallbacks, and required-tool recall.
* **System**: p50/p95 latency, tokens per query, cost (0 for self-hosted), API load test, and inference TTFT and tokens/s.
* **Controlled ablations**: each `configs/experiments/*.yaml` overrides only the variable under test. Same model, prompts, and benchmark.

```bash
make benchmark-retrieval          # 6 retrieval variants, ~1 min
make benchmark                    # + base LLM / extractive / vanilla / hybrid / hybrid+rerank RAG
make benchmark-agent              # agent policies; also writes evaluated trajectories
python scripts/run_benchmark.py configs/experiments/agent_learned.yaml
make results                      # regenerate docs/experiments/results.md
python scripts/make_readme_figures.py   # regenerate docs/assets charts
```

## Post-training and RL commands

```bash
make benchmark-agent                                   # collect evaluated trajectories
python -m atlas trajectories validate data/trajectories/agent_heuristic.jsonl
python scripts/train_sft.py -c configs/sft/lora.yaml   # LoRA SFT + base-vs-adapter eval (MPS/CPU/CUDA)
python scripts/train_sft.py -c configs/sft/qlora.yaml  # QLoRA, 7B, needs CUDA — configured, not validated
python scripts/train_rl.py -c configs/rl/reinforce.yaml   # router policy RL (CPU/MPS, minutes)
python scripts/train_grpo.py --smoke                   # GRPO pipeline check; full runs need a GPU
python scripts/run_benchmark.py configs/experiments/agent_sft.yaml configs/experiments/agent_learned.yaml
```

## Serving and inference benchmarks

```bash
python -m atlas serve-model --port 8001                # OpenAI-compatible HF server (vLLM stand-in)
python scripts/bench_inference.py --base-url http://127.0.0.1:8001/v1 --concurrency 4
docker compose -f docker/docker-compose.yml --profile gpu up   # vLLM + api + workers + redis + qdrant — configured, not validated
python scripts/bench_inference.py --base-url http://localhost:8001/v1 --model Qwen/Qwen2.5-7B-Instruct --concurrency 16
python scripts/run_system_benchmark.py --endpoint retrieve --concurrency 8 --requests 200
kubectl apply -k deployment/kubernetes                 # requires a cluster (+ KEDA, GPU nodes)
```

Single-node mode (`atlas serve`, in-memory queue, FAISS) is the default and is what the measurements used. Distributed mode switches config only: `ATLAS__SERVING__QUEUE_BACKEND=redis`, `ATLAS__RETRIEVAL__VECTOR_STORE=qdrant`, and `atlas worker` processes.

## Repository layout

```
src/atlas/
  common/        typed config (YAML + env), domain types, cache, safe paths, errors
  ingestion/     loaders (PDF/TXT/MD) and the ingest pipeline
  chunking/      sentence-packing chunker with section detection
  embeddings/    sentence-transformers + hashing embedders, disk embedding cache
  retrieval/     vector stores (NumPy/FAISS/Qdrant), BM25, fusion, filters, index, retriever
  reranking/     cross-encoder + lexical rerankers
  multimodal/    table/figure extraction, SQLite table store, VLM providers
  llm/           provider-agnostic client (HF local, OpenAI-compatible/vLLM, Anthropic, extractive)
  rag/           prompts, citation parsing/validation, RAG + closed-book pipelines
  agents/        LangGraph graph, nodes, policies, features/action mask, state, reducers
  tools/         tool interface/executor + documents, SQL, Python sandbox, visual, web tools
  memory/        session memory helpers, persistent research history
  evaluation/    benchmark schema, metrics, runners
  trajectory/    trajectory schema, JSONL store, validation, quality filter
  training/      trajectory → SFT data, LoRA/QLoRA trainer, imitation eval
  rl/            environment, reward, policy network, REINFORCE trainer, GRPO
  integrations/  LangChain adapters (retriever + tools) over the hand-written components
  serving/       FastAPI app, schemas, job store/queue, worker, dev model server, web UI
  observability/ structured logging, tracing, Prometheus metrics
configs/         default.yaml, experiments/, sft/, rl/
scripts/         benchmarks, training, demo, corpus download, results table, README figures
tests/           unit/ integration/ e2e/   (offline; `-m slow` for real models)
docker/, deployment/kubernetes/, docs/, benchmarks/results/
```

## Limitations

* The benchmark is small (46 questions, one domain). Most ablation differences fall inside the CIs.
* The generator is a local 1.5B model. Several errors are misread numbers even when the right evidence was retrieved, which is why better retrieval does not always improve end-to-end accuracy.
* Groundedness is a lexical proxy. Answer matching is containment-based, so a verbose answer that contains the gold span counts as correct.
* Table detection handles ruled tables only. The local 256M VLM reads charts poorly; use a 7B or hosted VLM.
* RL uses an extractive writer inside rollouts and a small train/test split. The learned router mainly learned *when to stop*, not when to use table or figure tools.
* LLM routers rarely decide to stop. The 1.5B router chose `answer` itself in 3 of 46 runs; the rest ended at the tool budget.
* SFT imitated those trajectories and trained on router examples only (a metadata bug dropped writer examples; fixed and regression-tested). It improved protocol-following but lowered end-to-end accuracy versus its 0.5B base. Next: filter out budget-forced steps and re-collect data with writer examples.
* Multimodal tools help only marginally here: table questions 37.5% → 50% with the table tools; compute questions 0% for all agents. The demo output (`docs/experiments/demo_output.md`) is unedited and includes wrong answers.
* vLLM, real Redis, Docker images, Kubernetes, QLoRA, and full GRPO runs were not executed here (no NVIDIA GPU, Docker daemon, or cluster). See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md). The next items are a larger benchmark with unanswerable and pixel-only questions, NLI and LLM judges, real vLLM measurements, and GRPO at scale.

## License

MIT. The corpus consists of public arXiv papers downloaded by script (not redistributed).
