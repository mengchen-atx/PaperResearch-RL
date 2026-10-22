# Atlas developer commands. Python env: uv (https://docs.astral.sh/uv/).
PY ?= .venv/bin/python
export HF_HUB_OFFLINE ?= 0

.PHONY: setup data ingest ask research run serve-model test test-all lint benchmark benchmark-retrieval \
        benchmark-agent benchmark-multimodal train-sft train-rl train-grpo-smoke results docker-up k8s-render clean \
        quickstart

setup:              ## create venv and install all extras
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[train,vision,dev]"

data:               ## download the public demo corpus and build/verify the benchmark
	bash scripts/download_corpus.sh data/raw
	$(PY) scripts/build_benchmark.py

ingest:             ## build the index (text + tables + figures)
	$(PY) -m atlas ingest data/raw

ask:                ## example RAG question
	$(PY) -m atlas ask "What is self-attention?"

research:           ## example multi-step research question
	$(PY) -m atlas research "By how many BLEU points does Transformer (big) outperform the ConvS2S Ensemble on EN-DE?"

run:                ## API + web UI on :8000
	$(PY) -m atlas serve --port 8000

serve-model:        ## OpenAI-compatible dev inference server on :8001 (vLLM stand-in)
	$(PY) -m atlas serve-model --port 8001

test:               ## unit + integration + e2e (offline, no model downloads)
	$(PY) -m pytest tests

test-all:           ## include real-model e2e test
	ATLAS_RUN_SLOW=1 $(PY) -m pytest tests

lint:
	$(PY) -m ruff check src tests scripts

benchmark-retrieval:
	$(PY) scripts/run_rag_benchmark.py retrieval

benchmark:          ## retrieval + generation ablation ladder
	$(PY) scripts/run_rag_benchmark.py all

benchmark-agent:    ## agent policies (+ collects evaluated trajectories)
	$(PY) scripts/run_agent_benchmark.py

benchmark-multimodal:
	$(PY) scripts/run_multimodal_benchmark.py

results:            ## regenerate docs/experiments/results.md from benchmarks/results
	$(PY) scripts/make_results_table.py

train-sft:
	$(PY) scripts/train_sft.py -c configs/sft/lora.yaml

train-rl:
	$(PY) scripts/train_rl.py -c configs/rl/reinforce.yaml

train-grpo-smoke:
	$(PY) scripts/train_grpo.py --smoke

docker-up:
	docker compose -f docker/docker-compose.yml up --build

k8s-render:         ## render manifests (requires kubectl)
	kubectl kustomize deployment/kubernetes

clean:
	rm -rf .pytest_cache .ruff_cache data/processed/cache

# CPU-only, no model downloads. Uses its own index dir so it never overwrites the real (bge) index.
QUICK_ENV = HF_HUB_OFFLINE=1 ATLAS__EMBEDDING__PROVIDER=hashing ATLAS__EMBEDDING__CACHE=false \
            ATLAS__LLM__PROVIDER=extractive ATLAS__LLM__MODEL=extractive ATLAS__RERANKER__ENABLED=false \
            ATLAS__VLM__PROVIDER=none ATLAS__AGENT__PLANNER=heuristic ATLAS__AGENT__CRITIC=heuristic \
            ATLAS__PATHS__INDEX_DIR=data/processed/quickstart_index \
            ATLAS__TOOLS__SQL_DB_PATH=data/processed/quickstart_index/tables.sqlite
quickstart:         ## CPU-only, no model downloads: corpus -> index -> one RAG answer -> one agent run
	bash scripts/download_corpus.sh data/raw
	$(QUICK_ENV) $(PY) -m atlas ingest data/raw
	$(QUICK_ENV) $(PY) -m atlas ask "What is the model dimension d_model used by the Transformer?"
	$(QUICK_ENV) $(PY) -m atlas research "By how many BLEU points does Transformer (big) outperform the ConvS2S Ensemble on EN-DE?"
