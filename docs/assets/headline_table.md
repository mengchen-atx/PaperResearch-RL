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
