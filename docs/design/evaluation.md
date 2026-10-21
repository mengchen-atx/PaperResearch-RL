# Design: Evaluation

* Benchmark schema: `evaluation/benchmark.py`; builder + verifier: `scripts/build_benchmark.py`.
* Metrics: `evaluation/metrics.py` (pure functions, unit-tested against hand-computed values).
* Runners: `evaluation/runner.py` (retrieval / RAG / agent), experiment configs in `configs/experiments/`.
* Output: `benchmarks/results/<experiment>/{config.json, records.jsonl, summary.json}`; `scripts/make_results_table.py`
  regenerates `docs/experiments/results.md` with bootstrap CIs.

Controlled comparisons: each experiment YAML overrides only the variables under test on top of
`configs/default.yaml` (e.g. `vanilla_rag` = dense, no reranker; `hybrid_rag` = + BM25 + RRF;
`hybrid_rerank_rag` = + cross-encoder), same LLM, same benchmark, same prompts.

Known limitations: 46 questions (small), a single corpus domain, lexical groundedness proxy, answer matching by
normalized containment (a verbose answer containing the gold span counts as correct).
