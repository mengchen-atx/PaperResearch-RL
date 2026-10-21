"""Composition root: builds every component from an `AtlasConfig`.

Components are created lazily (`cached_property`) so that, e.g., `atlas ask`
never loads the VLM and unit tests can inject lightweight fakes by overriding
single attributes. This is the only module that knows about concrete classes;
everything else depends on interfaces.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

from atlas.agents.features import Capabilities
from atlas.agents.graph import ResearchAgent
from atlas.agents.nodes import AgentContext
from atlas.agents.policies import HeuristicPolicy, LearnedPolicy, LLMPolicy, Policy
from atlas.common.cache import Cache, build_cache
from atlas.common.config import AtlasConfig
from atlas.common.types import Modality
from atlas.embeddings.embedder import Embedder, build_embedder
from atlas.ingestion.pipeline import IngestReport, ingest_paths
from atlas.llm.base import LLMClient
from atlas.llm.providers import build_llm
from atlas.memory.history import ResearchHistory
from atlas.multimodal.vlm import VisionLanguageModel, build_vlm
from atlas.observability.logging import get_logger
from atlas.rag.pipeline import BaseLLMPipeline, RAGPipeline
from atlas.reranking.reranker import build_reranker
from atlas.retrieval.index import KnowledgeIndex, load_index
from atlas.retrieval.retriever import Retriever
from atlas.tools.base import Tool, ToolExecutor
from atlas.tools.documents import ReadDocument, SearchDocuments
from atlas.tools.python_sandbox import RunPython
from atlas.tools.sql import LookupTable, QuerySQL, table_catalog
from atlas.tools.visual import AnalyzeImage
from atlas.tools.web import SearchWeb
from atlas.trajectory.store import TrajectoryStore

log = get_logger("runtime")


class AtlasRuntime:
    def __init__(self, cfg: AtlasConfig) -> None:
        self.cfg = cfg

    # --- paths ------------------------------------------------------------------------
    @property
    def index_dir(self) -> Path:
        return self.cfg.resolve_path(self.cfg.paths.index_dir)

    @property
    def table_db(self) -> Path:
        return self.cfg.resolve_path(self.cfg.tools.sql_db_path)

    # --- knowledge layer ----------------------------------------------------------------
    @cached_property
    def cache(self) -> Cache:
        return build_cache(self.cfg.cache, self.cfg.serving.redis_url)

    @cached_property
    def embedder(self) -> Embedder:
        return build_embedder(self.cfg.embedding, self.cfg.resolve_path(self.cfg.paths.cache_dir))

    @cached_property
    def index(self) -> KnowledgeIndex:
        return load_index(self.cfg, self.index_dir)

    @cached_property
    def retriever(self) -> Retriever:
        return Retriever(self.index, self.embedder, self.cfg.retrieval, build_reranker(self.cfg.reranker), self.cache)

    def ingest(self, sources: list[Path]) -> IngestReport:
        index, report = ingest_paths(self.cfg, self.embedder, sources, self.index_dir)
        self.__dict__["index"] = index
        self.__dict__.pop("retriever", None)
        self.__dict__.pop("executor", None)
        return report

    # --- models -------------------------------------------------------------------------
    @cached_property
    def llm(self) -> LLMClient:
        return build_llm(self.cfg.llm, self.cache)

    @cached_property
    def vlm(self) -> VisionLanguageModel:
        return build_vlm(self.cfg.vlm)

    # --- pipelines & agent --------------------------------------------------------------
    def rag(self) -> RAGPipeline:
        return RAGPipeline(self.retriever, self.llm, self.cfg.llm.max_new_tokens,
                           llm_query_rewrite=self.cfg.retrieval.query_rewrite == "llm")

    def base_llm(self) -> BaseLLMPipeline:
        return BaseLLMPipeline(self.llm, self.cfg.llm.max_new_tokens)

    def capabilities(self) -> Capabilities:
        has_figs = any(c.modality == Modality.FIGURE for c in self.index.chunks)
        return Capabilities(
            python=self.cfg.tools.python_mode == "sandbox",
            visual=self.cfg.vlm.provider != "none" and has_figs,
            web=self.cfg.tools.web_backend != "disabled",
            tables=bool(table_catalog(self.table_db)),
            llm_can_reason=self.cfg.llm.provider != "extractive",
        )

    def tools(self) -> list[Tool]:
        tools: list[Tool] = [SearchDocuments(self.retriever), ReadDocument(self.retriever),
                             QuerySQL(self.table_db), LookupTable(self.table_db), RunPython(self.cfg.tools),
                             SearchWeb(self.cfg.tools)]
        if self.cfg.vlm.provider != "none":
            roots = [self.index_dir / "figures", self.cfg.resolve_path(self.cfg.paths.upload_dir)]
            tools.append(_LazyVisualTool(self, roots))
        return tools

    @cached_property
    def executor(self) -> ToolExecutor:
        return ToolExecutor(self.tools(), timeout_s=self.cfg.tools.tool_timeout_s)

    def policy(self, name: str | None = None) -> Policy:
        name = name or self.cfg.agent.policy
        if name == "llm":
            return LLMPolicy(self.llm)
        if name == "learned":
            return LearnedPolicy(self.cfg.resolve_path(self.cfg.agent.learned_policy_path),
                                 self.cfg.agent.max_iterations, self.cfg.agent.max_tool_calls)
        return HeuristicPolicy()

    def agent(self, policy: str | None = None) -> ResearchAgent:
        ctx = AgentContext(cfg=self.cfg.agent, llm=self.llm, executor=self.executor, policy=self.policy(policy),
                           caps=self.capabilities(), top_k=self.cfg.retrieval.top_k,
                           max_new_tokens=self.cfg.llm.max_new_tokens)
        return ResearchAgent(ctx)

    # --- persistence --------------------------------------------------------------------
    @cached_property
    def history(self) -> ResearchHistory:
        return ResearchHistory(self.cfg.resolve_path(self.cfg.paths.data_dir) / "processed" / "history.sqlite")

    @cached_property
    def trajectories(self) -> TrajectoryStore:
        return TrajectoryStore(self.cfg.resolve_path(self.cfg.paths.trajectories_dir) / self.cfg.trajectory.filename)


class _LazyVisualTool(AnalyzeImage):
    """Defers loading the VLM until the agent actually needs visual analysis."""

    def __init__(self, rt: AtlasRuntime, roots: list[Path]) -> None:
        self._rt = rt
        self.retriever = rt.retriever
        self.allowed_roots = roots

    @property
    def vlm(self) -> VisionLanguageModel:  # type: ignore[override]
        return self._rt.vlm
