"""Typed, layered configuration.

Resolution order (later wins):
    1. configs/default.yaml
    2. optional override YAML(s) (e.g. an experiment config)
    3. environment variables of the form ATLAS__SECTION__KEY=value

Secrets are never stored in config. Provider sections reference the *name* of an
environment variable (`api_key_env`) and the key is read at call time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from atlas.common.errors import ConfigError
from atlas.common.paths import project_root

ENV_PREFIX = "ATLAS__"


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(Section):
    data_dir: str = "data"
    index_dir: str = "data/processed/index"
    trajectories_dir: str = "data/trajectories"
    runs_dir: str = "benchmarks/results"
    cache_dir: str = "data/processed/cache"
    trace_dir: str = "data/processed/traces"
    upload_dir: str = "data/raw/uploads"


class ChunkingConfig(Section):
    chunk_size: int = Field(1000, gt=50, description="Target chunk size in characters")
    chunk_overlap: int = Field(150, ge=0)
    min_chunk_chars: int = Field(40, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class EmbeddingConfig(Section):
    provider: Literal["sentence_transformers", "hashing"] = "sentence_transformers"
    model_name: str = "BAAI/bge-small-en-v1.5"
    # bge models retrieve better when queries carry this instruction prefix.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 32
    device: str = "auto"
    hashing_dim: int = 512
    cache: bool = True


class RetrievalConfig(Section):
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    vector_store: Literal["numpy", "faiss", "qdrant"] = "faiss"
    top_k: int = Field(5, gt=0)
    candidate_k: int = Field(30, gt=0)
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    dense_weight: float = Field(0.5, ge=0.0, le=1.0)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    query_rewrite: Literal["none", "keywords", "llm"] = "none"
    qdrant_url: str | None = None
    qdrant_path: str = "data/processed/qdrant"
    qdrant_collection: str = "atlas_chunks"


class RerankerConfig(Section):
    enabled: bool = False
    provider: Literal["cross_encoder", "lexical"] = "cross_encoder"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "auto"


class LLMConfig(Section):
    provider: Literal["hf_local", "openai_compatible", "anthropic", "extractive"] = "hf_local"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_new_tokens: int = 384
    device: str = "auto"
    timeout_s: float = 120.0
    max_retries: int = 2
    adapter_path: str | None = None  # LoRA adapter for hf_local
    # USD per 1M tokens, used for cost estimates. 0 for self-hosted models.
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0


class VLMConfig(Section):
    provider: Literal["hf_local", "openai_compatible", "anthropic", "none"] = "hf_local"
    model: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    base_url: str | None = None
    api_key_env: str | None = None
    max_new_tokens: int = 256
    device: str = "auto"
    timeout_s: float = 120.0


class MultimodalConfig(Section):
    extract_tables: bool = True
    extract_figures: bool = True
    describe_figures_at_ingest: bool = False
    min_image_pixels: int = 150 * 150


class AgentConfig(Section):
    policy: Literal["llm", "heuristic", "learned"] = "heuristic"
    learned_policy_path: str = "data/processed/rl/policy.pt"
    max_iterations: int = Field(4, gt=0)
    max_tool_calls: int = Field(10, gt=0)
    timeout_s: float = 300.0
    critic: Literal["llm", "heuristic"] = "heuristic"
    min_evidence: int = 2
    sufficient_score: float = 0.5
    planner: Literal["llm", "heuristic"] = "llm"
    max_subquestions: int = 3


class ToolsConfig(Section):
    python_mode: Literal["disabled", "sandbox"] = "sandbox"
    python_timeout_s: float = 5.0
    python_memory_mb: int = 256
    sql_db_path: str = "data/processed/index/tables.sqlite"
    web_backend: Literal["wikipedia", "tavily", "disabled"] = "disabled"
    web_max_results: int = 3
    web_timeout_s: float = 10.0
    tool_timeout_s: float = 60.0


class ServingConfig(Section):
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout_s: float = 300.0
    queue_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    max_concurrent_jobs: int = 2
    max_upload_mb: int = 50
    rate_limit_per_minute: int = 60


class CacheConfig(Section):
    backend: Literal["none", "memory", "redis"] = "memory"
    ttl_s: int = 3600
    max_items: int = 2048


class ObservabilityConfig(Section):
    log_level: str = "INFO"
    log_json: bool = False
    tracing: bool = True
    metrics: bool = True


class TrajectoryConfig(Section):
    enabled: bool = True
    filename: str = "trajectories.jsonl"


class AtlasConfig(Section):
    environment: Literal["development", "production"] = "development"
    paths: PathsConfig = PathsConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    llm: LLMConfig = LLMConfig()
    vlm: VLMConfig = VLMConfig()
    multimodal: MultimodalConfig = MultimodalConfig()
    agent: AgentConfig = AgentConfig()
    tools: ToolsConfig = ToolsConfig()
    serving: ServingConfig = ServingConfig()
    cache: CacheConfig = CacheConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    trajectory: TrajectoryConfig = TrajectoryConfig()

    @model_validator(mode="after")
    def _production_guards(self) -> AtlasConfig:
        # Arbitrary code execution must be an explicit opt-in outside development.
        override = os.environ.get("ATLAS_ALLOW_PYTHON_IN_PRODUCTION") == "1"
        if self.environment == "production" and self.tools.python_mode != "disabled" and not override:
            raise ValueError(
                "tools.python_mode must be 'disabled' in production "
                "(set ATLAS_ALLOW_PYTHON_IN_PRODUCTION=1 to override deliberately)"
            )
        return self

    def resolve_path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else project_root() / p

    def with_overrides(self, overrides: dict[str, Any]) -> AtlasConfig:
        """Return a new config with a nested dict of overrides applied (immutably)."""
        merged = deep_merge(self.model_dump(), overrides)
        return AtlasConfig.model_validate(merged)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_env_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]:
    environ = dict(os.environ) if environ is None else environ
    result: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX) :].split("__") if p]
        if not parts:
            continue
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_env_value(raw)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at the top level")
    # Experiment files may carry metadata sections that are not runtime config.
    return {k: v for k, v in data.items() if k not in {"experiment", "description"}}


def load_config(*override_files: str | Path, environ: dict[str, str] | None = None) -> AtlasConfig:
    merged: dict[str, Any] = {}
    default = project_root() / "configs" / "default.yaml"
    if default.exists():
        merged = _read_yaml(default)
    for f in override_files:
        merged = deep_merge(merged, _read_yaml(Path(f)))
    merged = deep_merge(merged, env_overrides(environ))
    try:
        return AtlasConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration:\n{exc}") from exc


def read_secret(env_name: str | None) -> str | None:
    """Read a secret from the environment by variable name. Never logged."""
    if not env_name:
        return None
    value = os.environ.get(env_name)
    return value or None
