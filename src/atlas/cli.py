"""Atlas command-line interface.

    atlas ingest data/raw                    build the index (text + tables + figures)
    atlas ask "What is self-attention?"      single-shot RAG with citations
    atlas retrieve "BLEU EN-DE" --mode bm25  inspect retrieval only
    atlas research "..." [--policy llm]      LangGraph research agent
    atlas serve                              FastAPI server
    atlas worker                             Redis queue worker
    atlas serve-model                        OpenAI-compatible dev inference server
    atlas trajectories validate|filter       trajectory data utilities

Global option `-c/--config FILE` layers an override YAML (e.g. an experiment).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from atlas.common.config import AtlasConfig, load_config
from atlas.common.errors import AtlasError
from atlas.observability.logging import configure_logging


def _runtime(cfg: AtlasConfig):
    from atlas.runtime import AtlasRuntime

    return AtlasRuntime(cfg)


def _print_evidence(items, show_scores: bool = True) -> None:
    for i, item in enumerate(items, start=1):
        chunk = getattr(item, "chunk", None)
        label = chunk.citation_label() if chunk else item.citation_label()
        text = chunk.text if chunk else item.content
        score = f" score={item.score:.4f}" if show_scores and item.score is not None else ""
        retr = f" via={item.retriever}" if chunk else f" via={item.source_tool}"
        section = f" section='{chunk.section}'" if chunk and chunk.section else ""
        print(f"  [{i}] {label}{score}{retr}{section}")
        print(textwrap.indent(textwrap.shorten(text.replace("\n", " "), 260), "      "))


def cmd_ingest(cfg: AtlasConfig, args) -> int:
    rt = _runtime(cfg)
    report = rt.ingest([Path(p) for p in args.paths])
    print(f"Ingested: {report.summary()}")
    print(f"Index written to {rt.index_dir}")
    return 0


def cmd_ask(cfg: AtlasConfig, args) -> int:
    rt = _runtime(cfg)
    pipe = rt.base_llm() if args.no_retrieval else rt.rag()
    res = pipe.answer(args.question, top_k=args.top_k)
    if args.json:
        print(res.model_dump_json(indent=2))
        return 0
    print(f"\nQuestion: {res.question}\n\nAnswer ({res.model}):\n{textwrap.fill(res.answer, 100)}\n")
    if res.invalid_citations:
        print(f"WARNING: answer cited non-existent evidence {res.invalid_citations}\n")
    if res.retrieved:
        print("Retrieved evidence:")
        _print_evidence(res.retrieved)
    print(f"\nlatency: retrieval {res.retrieval_ms:.0f} ms, generation {res.generation_ms:.0f} ms; "
          f"tokens: {res.usage.prompt_tokens} in / {res.usage.completion_tokens} out")
    return 0


def cmd_retrieve(cfg: AtlasConfig, args) -> int:
    overrides = {"retrieval": {"mode": args.mode}} if args.mode else {}
    if args.rerank:
        overrides["reranker"] = {"enabled": True}
    rt = _runtime(cfg.with_overrides(overrides))
    from atlas.retrieval.filters import MetadataFilter

    flt = MetadataFilter(modalities=[args.modality]) if args.modality else None
    _print_evidence(rt.retriever.retrieve(args.query, top_k=args.top_k, flt=flt))
    return 0


def cmd_research(cfg: AtlasConfig, args) -> int:
    rt = _runtime(cfg)
    agent = rt.agent(args.policy)
    result = None
    for node, update in agent.stream(args.question):
        if node == "final":
            result = update["result"]
        elif not args.json:
            _print_progress(node, update)
    assert result is not None
    if cfg.trajectory.enabled:
        rt.trajectories.append(result.to_trajectory())
    rt.history.record(result.session_id, result.query, result.answer,
                      [c.model_dump() for c in result.citations], len(result.tool_history), result.status)
    if args.json:
        print(result.model_dump_json(indent=2))
        return 0
    print("\n" + "=" * 100)
    print(f"RESEARCH REPORT  (policy={result.policy}, status={result.status}, trace={result.trace_id})")
    print("=" * 100)
    print("Plan:")
    for sq in result.subquestions:
        print(f"  - {sq}")
    print(f"\nAnswer:\n{textwrap.fill(result.answer, 100)}\n")
    print("Sources:")
    for s in result.report.get("sources", []):
        flag = "*" if s["cited"] else " "
        print(f" {flag}[{s['marker']}] {s['label']}  ({s['modality']} via {s['tool']})")
    print(f"\n{len(result.tool_history)} tool calls, {result.iterations} iterations, "
          f"{result.prompt_tokens + result.completion_tokens} LLM tokens, {result.latency_ms / 1000:.1f}s")
    return 0


def _print_progress(node: str, update: dict) -> None:
    if node == "planner" and update.get("subquestions"):
        print(f"[planner] {len(update['subquestions'])} sub-question(s): {update['subquestions']}")
    elif node == "router" and update.get("next_action"):
        a = update["next_action"]
        fb = " (fallback)" if a.get("fallback") else ""
        print(f"[router]  -> {a['action']}{fb}: {a.get('rationale', '')[:90]}")
    elif node == "act":
        for rec in update.get("tool_history", []):
            status = "ok" if rec.ok else f"error: {rec.error}"
            print(f"[tool]    {rec.tool}({json.dumps(rec.arguments)[:90]}) -> {status}, "
                  f"+{rec.new_evidence} evidence, {rec.latency_ms:.0f} ms")
    elif node == "critic":
        for c in update.get("critiques", []):
            print(f"[critic]  coverage={c['coverage']:.2f} sufficient={c['sufficient']} all_resolved={c['all_resolved']}")
    elif node == "writer":
        print("[writer]  report written")


def cmd_serve(cfg: AtlasConfig, args) -> int:
    import uvicorn

    from atlas.serving.api import create_app

    uvicorn.run(create_app(cfg), host=args.host or cfg.serving.host, port=args.port or cfg.serving.port,
                log_level=cfg.observability.log_level.lower())
    return 0


def cmd_worker(cfg: AtlasConfig, args) -> int:
    from atlas.serving.worker import run_worker

    run_worker(cfg)
    return 0


def cmd_serve_model(cfg: AtlasConfig, args) -> int:
    import uvicorn

    from atlas.serving.model_server import create_model_app

    uvicorn.run(create_model_app(cfg.llm), host="127.0.0.1", port=args.port)
    return 0


def cmd_trajectories(cfg: AtlasConfig, args) -> int:
    from atlas.trajectory.store import TrajectoryStore, quality_filter, validate_file

    path = Path(args.path)
    if args.action == "validate":
        print(json.dumps(validate_file(path), indent=2))
        return 0
    keep = quality_filter(min_score=args.min_score)
    trajs = [t for t in TrajectoryStore(path).iter() if keep(t)]
    n = TrajectoryStore(Path(args.out)).write_all(trajs)
    print(f"kept {n} trajectories -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atlas", description="Atlas multimodal agentic research system")
    p.add_argument("-c", "--config", action="append", default=[], help="override YAML (repeatable)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="ingest documents and build the index")
    s.add_argument("paths", nargs="+")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("ask", help="single-shot RAG question")
    s.add_argument("question")
    s.add_argument("-k", "--top-k", type=int, default=None)
    s.add_argument("--no-retrieval", action="store_true", help="closed-book base LLM")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("retrieve", help="retrieval only")
    s.add_argument("query")
    s.add_argument("-k", "--top-k", type=int, default=5)
    s.add_argument("--mode", choices=["dense", "bm25", "hybrid"])
    s.add_argument("--rerank", action="store_true")
    s.add_argument("--modality", choices=["text", "table", "figure"])
    s.set_defaults(fn=cmd_retrieve)

    s = sub.add_parser("research", help="run the research agent")
    s.add_argument("question")
    s.add_argument("--policy", choices=["heuristic", "llm", "learned"])
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_research)

    s = sub.add_parser("serve", help="run the API server")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("worker", help="run a queue worker (requires Redis)")
    s.set_defaults(fn=cmd_worker)

    s = sub.add_parser("serve-model", help="OpenAI-compatible dev inference server (HF transformers)")
    s.add_argument("--port", type=int, default=8001)
    s.set_defaults(fn=cmd_serve_model)

    s = sub.add_parser("trajectories", help="validate or filter trajectory JSONL")
    s.add_argument("action", choices=["validate", "filter"])
    s.add_argument("path")
    s.add_argument("--out", default="data/trajectories/filtered.jsonl")
    s.add_argument("--min-score", type=float, default=0.5)
    s.set_defaults(fn=cmd_trajectories)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(*args.config)
        configure_logging("DEBUG" if args.verbose else cfg.observability.log_level, cfg.observability.log_json)
        return args.fn(cfg, args)
    except AtlasError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
