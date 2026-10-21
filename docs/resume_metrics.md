# Resume metrics: what can be claimed, and where each number comes from

Rule: **a number goes on a resume only if it is produced by a command in this repo and is present in a results
file.** Never round in your favor, always state the benchmark size, and prefer "on a 46-question verified
benchmark" over unqualified percentages. Current measured values live in
[`docs/experiments/results.md`](experiments/results.md) (auto-generated) — copy from there, not from memory.

| Metric | Definition | Produced by | Source file | Status |
|---|---|---|---|---|
| Recall@K (retrieval) | fraction of gold evidence pages in top-K | `python scripts/run_rag_benchmark.py retrieval` | `benchmarks/results/retrieval_*/summary.json` | measured |
| MRR / nDCG@K | rank quality of first/all relevant results | same | same | measured |
| Answer accuracy | numeric match (tolerance) or boundary-aware gold containment | `python scripts/run_rag_benchmark.py generation` | `benchmarks/results/{base_llm,vanilla_rag,hybrid_rag,hybrid_rerank_rag}/records.jsonl` | measured |
| Δ accuracy vs closed-book | RAG accuracy − base-LLM accuracy, same model | same | same | measured |
| Citation validity | share of citation markers pointing to real evidence | same | records.jsonl (`citation_validity`) | measured |
| Citation precision | share of cited evidence located on a gold page | same | records.jsonl (`citation_precision`) | measured |
| Groundedness (lexical proxy) | share of answer sentences supported by evidence tokens | same | records.jsonl | measured (proxy — say "lexical") |
| Agent task success | answer accuracy of the LangGraph agent | `python scripts/run_agent_benchmark.py` | `benchmarks/results/agent_*/records.jsonl` | measured |
| Tool calls / task, failed & unproductive calls | from agent tool history | same | same | measured |
| Required-tool recall | table/figure/compute questions that used the matching tool | same | same | measured |
| RL: tool calls & success vs heuristic/random | held-out episodes in the research env | `python scripts/train_rl.py` | `data/processed/rl/results.json` | measured (small n) |
| SFT: router action accuracy base vs LoRA | held-out trajectory steps | `python scripts/train_sft.py` | `data/processed/sft/eval.json` | measured (small n) |
| LoRA trainable-parameter share | trainable / total params | same | `data/processed/sft/train_metrics.json` | measured |
| p50 / p95 latency | per-query wall time | every benchmark | `summary.json` | measured on Apple M4 (MPS) |
| API throughput & tail latency | HTTP load test | `python scripts/run_system_benchmark.py` | `benchmarks/results/system/*.json` | measured locally |
| TTFT, decode tokens/s, req/s | streaming inference benchmark | `python scripts/bench_inference.py` | `benchmarks/results/inference/*.json` | measured on HF dev server; **vLLM numbers pending a GPU host** |
| Tokens / query, cost / query | LLM usage × configured price | benchmarks | records.jsonl | measured tokens; cost = 0 for self-hosted (set prices for API models) |
| GPU memory | peak VRAM for serving/training | not collected (no NVIDIA GPU) | – | **pending** |
| vLLM throughput vs HF baseline | same benchmark against vLLM | `bench_inference.py --base-url http://<vllm>:8000/v1` | – | **pending (requires CUDA GPU)** |
| QLoRA 7B training | memory/time/quality | `python scripts/train_sft.py -c configs/sft/qlora.yaml` | – | **pending (requires CUDA GPU)** |
| Kubernetes scaling behaviour | throughput vs replicas | `kubectl apply -k deployment/kubernetes` + load test | – | **pending (requires a cluster)** |

## How to phrase small-sample results
* "Hybrid retrieval raised Recall@5 from 81.5% to 87.0% on a 46-question, page-labeled benchmark" — fine.
* "RAG improved accuracy by 5×" — do not compress a 10.9% → ~60% change into a multiplier without the base rate.
* RL / SFT results come from ≈13 held-out questions / a few dozen trajectory steps: describe them as pipeline
  validation with a directional result, e.g. "matched the heuristic policy's success rate with 41% fewer tool
  calls on held-out questions", not as a general improvement.
