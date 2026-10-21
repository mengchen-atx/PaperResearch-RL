"""Benchmark runners: retrieval, RAG (incl. base LLM), and agent.

Each runner produces per-item records (JSON-serializable dicts) and a summary.
`run_experiment` executes one experiment config end-to-end and writes:

    benchmarks/results/<experiment>/
        config.json    fully-resolved AtlasConfig + experiment metadata
        records.jsonl  one record per benchmark item
        summary.json   aggregate metrics (+ by question type), environment info
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from atlas.agents.actions import Action
from atlas.agents.graph import ResearchAgent
from atlas.common.config import AtlasConfig, load_config
from atlas.evaluation import metrics as M
from atlas.evaluation.benchmark import BenchmarkItem, load_benchmark
from atlas.observability.logging import get_logger
from atlas.rag.pipeline import BaseLLMPipeline, RAGPipeline
from atlas.retrieval.retriever import Retriever
from atlas.trajectory.store import TrajectoryStore

log = get_logger("eval")
RETRIEVAL_KS = (1, 3, 5, 10)
_MODALITY_TOOL = {"table": {"lookup_table", "query_sql"}, "figure": {"analyze_image"},
                  "computation": {"run_python"}}


# ---------------------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------------------
def retrieval_record(item: BenchmarkItem, locs: list[M.Location], ks=RETRIEVAL_KS) -> dict[str, Any]:
    rec: dict[str, Any] = {"mrr": M.reciprocal_rank(locs, item.evidence)}
    for k in ks:
        rec[f"recall@{k}"] = M.recall_at_k(locs, item.evidence, k)
        rec[f"precision@{k}"] = M.precision_at_k(locs, item.evidence, k)
        rec[f"ndcg@{k}"] = M.ndcg_at_k(locs, item.evidence, k)
    return rec


def evaluate_retrieval(retriever: Retriever, items: list[BenchmarkItem], k_max: int = 10) -> list[dict]:
    records = []
    for item in items:
        t0 = time.perf_counter()
        hits = retriever.retrieve(item.question, top_k=k_max)
        latency = (time.perf_counter() - t0) * 1000
        locs = [(h.chunk.filename, h.chunk.page) for h in hits]
        records.append({"id": item.id, "type": item.type, "latency_ms": latency,
                        "retrieved": [f"{f}:p{p}" for f, p in locs], **retrieval_record(item, locs)})
    return records


# ---------------------------------------------------------------------------------------
# RAG / base LLM
# ---------------------------------------------------------------------------------------
def answer_record(item: BenchmarkItem, answer: str, evidence_texts: list[str],
                  evidence_locs: list[M.Location]) -> dict[str, Any]:
    return {
        "correct": float(M.answer_correct(answer, item)),
        "token_f1": M.token_f1(answer, item.answer),
        "groundedness": M.groundedness(answer, evidence_texts) if evidence_texts else 0.0,
        **M.citation_metrics(answer, evidence_locs, item.evidence),
    }


def evaluate_rag(pipeline: RAGPipeline | BaseLLMPipeline, items: list[BenchmarkItem], cost_fn=None) -> list[dict]:
    records = []
    for n, item in enumerate(items, start=1):
        res = pipeline.answer(item.question)
        locs = [(e.filename, e.page) for e in res.evidence]
        rec = {"id": item.id, "type": item.type, "question": item.question, "gold": item.answer,
               "answer": res.answer, "evidence_locs": locs, "latency_ms": res.total_ms, "retrieval_ms": res.retrieval_ms,
               "generation_ms": res.generation_ms, "prompt_tokens": res.usage.prompt_tokens,
               "completion_tokens": res.usage.completion_tokens,
               "cost_usd": cost_fn(res.usage) if cost_fn else 0.0,
               "invalid_citations": res.invalid_citations,
               **answer_record(item, res.answer, [e.content for e in res.evidence], locs)}
        if res.evidence:
            rec.update(retrieval_record(item, locs, ks=(1, 3, 5)))
        records.append(rec)
        log.info("rag item", extra={"n": n, "id": item.id, "correct": rec["correct"],
                                    "ms": round(res.total_ms)})
    return records


# ---------------------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------------------
def agent_metrics(item: BenchmarkItem, result) -> dict[str, Any]:
    history = result.tool_history
    tools_used = {r.tool for r in history if r.ok}
    required = [m for m in item.modalities if m in _MODALITY_TOOL]
    hits = [m for m in required if _MODALITY_TOOL[m] & tools_used]
    decisions = result.decisions
    return {
        "tool_calls": len(history),
        "failed_tool_calls": sum(1 for r in history if not r.ok),
        "unproductive_tool_calls": sum(1 for r in history if r.ok and r.new_evidence == 0),
        "duplicate_calls_skipped": sum(1 for r in history if r.error and "duplicate" in r.error),
        "trajectory_length": len(decisions),
        "completed": float(result.status in ("answered", "budget_exhausted") and bool(result.answer.strip())),
        "budget_exhausted": float(result.status in ("budget_exhausted", "timeout")),
        "replans": sum(1 for d in decisions if d["action"] == Action.REPLAN),
        "policy_fallbacks": sum(1 for d in decisions if d.get("fallback")),
        "required_tool_recall": (len(hits) / len(required)) if required else None,
        "tools_used": sorted(tools_used),
    }


def evaluate_agent(agent: ResearchAgent, items: list[BenchmarkItem], trajectories: TrajectoryStore | None = None,
                   cost_fn=None) -> list[dict]:
    records = []
    for n, item in enumerate(items, start=1):
        result = agent.run(item.question)
        evidence = result.evidence
        cited_ids = {c.evidence_id for c in result.citations}
        # [n] markers index into the writer's selected evidence == report["sources"]
        locs = [(s.get("filename"), s.get("page")) for s in result.report.get("sources", [])]
        ans = answer_record(item, result.answer, [e.content for e in evidence], locs)
        all_locs = [(e.filename, e.page) for e in evidence]
        rec = {"id": item.id, "type": item.type, "question": item.question, "gold": item.answer,
               "answer": result.answer, "status": result.status, "latency_ms": result.latency_ms,
               "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
               "cost_usd": 0.0, "evidence_count": len(evidence), "cited": len(cited_ids),
               "evidence_recall": M.recall_at_k(all_locs, item.evidence, len(all_locs) or 1),
               **ans, **agent_metrics(item, result)}
        if cost_fn:
            from atlas.common.types import TokenUsage

            rec["cost_usd"] = cost_fn(TokenUsage(prompt_tokens=result.prompt_tokens,
                                                 completion_tokens=result.completion_tokens))
        if trajectories is not None:
            base = result.to_trajectory()
            traj = base.model_copy(update={
                "evaluation": {"answer_score": rec["correct"], "token_f1": rec["token_f1"],
                               "citation_precision": rec["citation_precision"], "benchmark_id": item.id},
                "success": bool(rec["correct"]) and rec["has_valid_citation"] > 0,
                # merge, not replace: keeps writer_evidence_ids needed to rebuild writer SFT examples
                "metadata": {**base.metadata, "benchmark_id": item.id, "type": item.type}})
            trajectories.append(traj)
        records.append(rec)
        log.info("agent item", extra={"n": n, "id": item.id, "correct": rec["correct"],
                                      "tools": rec["tool_calls"], "ms": round(result.latency_ms)})
    return records


# ---------------------------------------------------------------------------------------
# Summaries & experiment orchestration
# ---------------------------------------------------------------------------------------
def summarize(records: list[dict]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    numeric_keys = sorted({k for r in records for k, v in r.items()
                           if isinstance(v, int | float) and not isinstance(v, bool)})
    summary: dict[str, Any] = {"n": len(records)}
    for k in numeric_keys:
        vals = [float(r[k]) for r in records if isinstance(r.get(k), int | float)]
        summary[k] = M.mean(vals)
    lat = [r["latency_ms"] for r in records if "latency_ms" in r]
    summary["latency_p50_ms"] = M.percentile(lat, 50)
    summary["latency_p95_ms"] = M.percentile(lat, 95)
    total_s = sum(lat) / 1000
    summary["throughput_qps_sequential"] = len(lat) / total_s if total_s else 0.0
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_type[r.get("type", "all")].append(r)
    key = "correct" if "correct" in records[0] else "recall@5"
    summary["by_type"] = {t: {"n": len(rs), key: M.mean([float(r.get(key, 0)) for r in rs])}
                          for t, rs in sorted(by_type.items())}
    return summary


def environment_info(cfg: AtlasConfig) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    from atlas.common.device import resolve_device

    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "device": resolve_device(cfg.llm.device), "git_commit": commit,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


def read_experiment(path: Path) -> dict[str, Any]:
    meta = (yaml.safe_load(path.read_text()) or {}).get("experiment") or {}
    if "name" not in meta or "system" not in meta:
        raise ValueError(f"{path}: experiment.name and experiment.system are required")
    return meta


def run_experiment(config_path: Path, benchmark_path: Path, out_root: Path, limit: int | None = None,
                   types: list[str] | None = None, collect_trajectories: bool = False) -> dict[str, Any]:
    from atlas.runtime import AtlasRuntime

    meta = read_experiment(config_path)
    cfg = load_config(config_path)
    rt = AtlasRuntime(cfg)
    items = load_benchmark(benchmark_path, types=types, limit=limit)
    system = meta["system"]
    t0 = time.perf_counter()
    cost_fn = None
    if system == "retrieval":
        records = evaluate_retrieval(rt.retriever, items)
    elif system == "base_llm":
        cost_fn = rt.llm.cost_usd
        records = evaluate_rag(rt.base_llm(), items, cost_fn)
    elif system == "rag":
        cost_fn = rt.llm.cost_usd
        records = evaluate_rag(rt.rag(), items, cost_fn)
    elif system == "agent":
        store = None
        if collect_trajectories:
            store = TrajectoryStore(cfg.resolve_path(cfg.paths.trajectories_dir) / f"{meta['name']}.jsonl")
        records = evaluate_agent(rt.agent(meta.get("policy")), items, store, rt.llm.cost_usd)
    else:
        raise ValueError(f"unknown experiment system '{system}'")

    summary = {"experiment": meta, "benchmark": str(benchmark_path), "num_items": len(items),
               "wall_time_s": time.perf_counter() - t0, "metrics": summarize(records),
               "models": {"llm": cfg.llm.model if system != "retrieval" else None,
                          "llm_provider": cfg.llm.provider, "embedding": cfg.embedding.model_name,
                          "reranker": cfg.reranker.model_name if cfg.reranker.enabled else None},
               "retrieval": {"mode": cfg.retrieval.mode, "fusion": cfg.retrieval.fusion,
                             "top_k": cfg.retrieval.top_k, "candidate_k": cfg.retrieval.candidate_k,
                             "vector_store": cfg.retrieval.vector_store},
               "environment": environment_info(cfg)}
    out = out_root / meta["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"experiment": meta, "config": cfg.model_dump(mode="json")},
                                                indent=2))
    with (out / "records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
