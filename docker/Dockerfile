# Atlas application image (API + worker share one image; command selects the role).
# CPU PyTorch keeps the image small; heavy LLM inference is delegated to the vLLM service.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf ATLAS__OBSERVABILITY__LOG_JSON=true
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Dependency layer first for better build caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install ".[vision]"

COPY configs ./configs
COPY scripts ./scripts

# Non-root runtime user.
RUN useradd --create-home --uid 10001 atlas && mkdir -p /app/data /models && chown -R atlas /app /models
USER atlas

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["atlas", "serve", "--host", "0.0.0.0", "--port", "8000"]
