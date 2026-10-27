# Contributing

Issues and pull requests are welcome. Please keep claims as narrow as the measurement that supports them.

## Setup

CPU path, no model downloads. Run from a fresh clone:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
make quickstart PY=python
pytest tests -m "not slow"
```

`make quickstart` always forces the hashing embedder and the extractive answerer (`Makefile` `QUICK_ENV`). It does not reproduce the Qwen benchmark numbers. CI does not run `make quickstart`.

Full local setup, including training extras, is `make setup` (requires [uv](https://docs.astral.sh/uv/) and Python 3.12). After that, `make quickstart` with no `PY=` uses `.venv/bin/python`.

## Checks

These match [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

```bash
ruff check src tests scripts
pytest tests -m "not slow"
```

`make lint` is the ruff command. `make test` is `pytest tests` with no marker filter. The one real-model test is marked `slow` and also skipped unless `ATLAS_RUN_SLOW=1`, so `make test` and `pytest tests -m "not slow"` run the same suite. `make test-all` sets `ATLAS_RUN_SLOW=1`.

## Benchmark numbers

Do not type metrics into the README or `docs/experiments/results.md`. Regenerate them:

```bash
make results                              # docs/experiments/results.md
python scripts/make_readme_figures.py     # docs/assets/*.png and architecture.svg
```

If a number is not in `benchmarks/results/` or `data/processed/`, do not add it. Configured-but-not-run components (vLLM, QLoRA, Kubernetes, a built Docker image) stay labeled that way. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## What not to commit

Corpus PDFs, indexes, model weights, and training artifacts are gitignored (`data/raw/`, `data/processed/`, `*.safetensors`). Do not commit API keys. Secrets are read from environment variables named in config, not stored in YAML.
