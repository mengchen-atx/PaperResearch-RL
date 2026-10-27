# Changelog

## 0.1.0

Initial public snapshot of Atlas. This version is the `version` in `pyproject.toml`. It is not a GitHub release until that tag is created.

- Hybrid retrieval (hand-written BM25, dense FAISS/NumPy, RRF or weighted fusion, optional cross-encoder) and a LangGraph agent (`planner → router → act → critic → writer`) with page-level citations.
- 46-question benchmark whose gold quotes are checked against the source PDFs. Committed results, with 95% bootstrap intervals, live in `benchmarks/results/` and `docs/experiments/results.md`.
- LoRA SFT on filtered agent trajectories and a REINFORCE router, both measured at small scale. The learned router matches a fixed single-search baseline on this benchmark. SFT improved protocol-following and did not improve end-to-end accuracy.
- FastAPI service (sync, async jobs, SSE) measured locally. vLLM, QLoRA, real Redis, Docker image builds, and Kubernetes were configured and not run.
- CPU quickstart (`make quickstart`) uses a hashing embedder and an extractive answerer so a clone can run without model downloads.
