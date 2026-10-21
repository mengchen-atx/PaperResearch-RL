"""LangGraph node implementations: planner, router, act (tools), critic, writer.

Each node reads the state and returns a *partial update*; it never mutates the
state in place. Loops are bounded three ways: `max_iterations` (router visits),
`max_tool_calls`, and a wall-clock deadline — plus LangGraph's recursion limit
as a last-resort backstop.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from atlas.agents.actions import ACTION_TOOL, Action, Decision, query_cues, term_coverage
from atlas.agents.features import Capabilities, current_subquestion
from atlas.agents.policies import HeuristicPolicy, Policy
from atlas.agents.prompts import WRITER_INSTRUCTIONS, critic_messages, planner_messages, python_messages
from atlas.agents.state import ResearchState
from atlas.common.config import AgentConfig
from atlas.common.types import Evidence, Modality
from atlas.llm.base import LLMClient, LLMResponse
from atlas.llm.json_utils import parse_json_object
from atlas.memory.session import call_signature, merge_evidence
from atlas.observability.logging import get_logger
from atlas.observability.tracing import get_tracer
from atlas.rag.citations import resolve_citations
from atlas.rag.prompts import grounded_messages
from atlas.retrieval.retriever import keyword_rewrite
from atlas.tools.base import ToolCallRecord, ToolExecutor

log = get_logger("agent")

MAX_WRITER_EVIDENCE = 8
EXHAUSTED_ATTEMPTS = 2
_SPLIT_QUESTION = re.compile(r"\?\s+|;\s+|\.\s+(?=(?:what|how|which|who|when|where|why|does|is|are)\b)", re.I)


@dataclass
class AgentContext:
    cfg: AgentConfig
    llm: LLMClient
    executor: ToolExecutor
    policy: Policy
    caps: Capabilities
    top_k: int = 5
    max_new_tokens: int = 384


def _llm_record(node: str, resp: LLMResponse) -> dict[str, Any]:
    return {"node": node, "model": resp.model, "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens, "latency_ms": round(resp.latency_ms, 1),
            "cached": resp.cached}


# ------------------------------------------------------------------------------------------
# Planner
# ------------------------------------------------------------------------------------------
def heuristic_plan(query: str, max_n: int) -> list[str]:
    parts = [p.strip(" ?.") for p in _SPLIT_QUESTION.split(query) if p and len(p.split()) >= 2]
    return [p + "?" for p in parts[:max_n]] if len(parts) > 1 else [query]


def _reformulate(subquestion: str, query: str) -> str:
    """Keyword form of a sub-question enriched with the original query's terms (deduplicated)."""
    terms = list(dict.fromkeys(keyword_rewrite(f"{subquestion} {query}").split()))
    return " ".join(terms)


def planner_node(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    query = state["query"]
    replanning = state.get("replans", 0) > 0 and bool(state.get("subquestions"))
    old = state.get("subquestions") or []
    resolved_idx = set(state.get("resolved") or [])
    missing = ""
    if replanning:
        crits = state.get("critiques") or []
        missing = crits[-1].get("missing", "") if crits else ""
    llm_calls: list[dict] = []
    subs: list[str] = []
    rationale = ""
    with get_tracer().span("agent.planner", kind="agent_node", replanning=replanning):
        if ctx.cfg.planner == "llm" and ctx.caps.llm_can_reason:
            resp = ctx.llm.generate(planner_messages(query, ctx.cfg.max_subquestions, missing), max_tokens=200,
                                    purpose="planner")
            llm_calls.append(_llm_record("planner", resp))
            parsed = parse_json_object(resp.text) or {}
            raw = parsed.get("subquestions")
            if isinstance(raw, list):
                subs = [str(s).strip() for s in raw if isinstance(s, str) and len(s.strip()) > 5]
                subs = subs[: ctx.cfg.max_subquestions]
            rationale = str(parsed.get("rationale", ""))[:500]
        if not subs:
            if replanning:
                # Heuristic reformulation: keyword form of each unresolved sub-question + the full query.
                subs = [_reformulate(old[i], query) for i in range(len(old)) if i not in resolved_idx] \
                    or [keyword_rewrite(query)]
            else:
                subs = heuristic_plan(query, ctx.cfg.max_subquestions)
            rationale = rationale or ("heuristic reformulation" if replanning else "heuristic decomposition")

    if replanning:
        kept = [old[i] for i in sorted(resolved_idx)]
        return {"subquestions": kept + subs, "resolved": list(range(len(kept))), "plan_rationale": rationale,
                "llm_calls": llm_calls, "sufficient": False}
    return {"subquestions": subs, "resolved": [], "plan_rationale": rationale, "llm_calls": llm_calls}


# ------------------------------------------------------------------------------------------
# Router
# ------------------------------------------------------------------------------------------
def budget_exhausted(state: ResearchState, cfg: AgentConfig) -> str | None:
    if state.get("iteration", 0) >= cfg.max_iterations:
        return "max_iterations"
    if len(state.get("tool_history") or []) >= cfg.max_tool_calls:
        return "max_tool_calls"
    if time.time() > state.get("deadline", float("inf")):
        return "timeout"
    return None


def router_node(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    with get_tracer().span("agent.router", kind="agent_node", policy=ctx.policy.name) as attrs:
        reason = budget_exhausted(state, ctx.cfg)
        llm_calls: list[dict] = []
        if reason:
            decision = Decision(action=Action.ANSWER, policy="budget", rationale=f"budget exhausted: {reason}")
        else:
            try:
                decision = ctx.policy.decide(state, ctx.caps)
                if decision.llm_call:
                    llm_calls = [{"node": "router", **decision.llm_call}]
            except Exception as exc:  # a failing policy (e.g. LLM outage) must not kill the run
                log.warning("policy failed; using heuristic", extra={"error": str(exc)})
                decision = HeuristicPolicy().decide(state, ctx.caps).model_copy(
                    update={"fallback": True, "rationale": f"policy error: {exc}"})
        attrs.update(action=decision.action.value, fallback=decision.fallback)
    update: dict[str, Any] = {
        "next_action": decision.model_dump(mode="json"),
        "decisions": [{**decision.model_dump(mode="json"), "iteration": state.get("iteration", 0),
                       "subquestion": current_subquestion(state)}],
        "iteration": state.get("iteration", 0) + 1,
        "llm_calls": llm_calls,
    }
    if reason:
        update["status"] = "timeout" if reason == "timeout" else "budget_exhausted"
    if decision.action == Action.REPLAN:
        update["replans"] = state.get("replans", 0) + 1
    return update


def route_after_router(state: ResearchState) -> str:
    action = (state.get("next_action") or {}).get("action")
    if action == Action.ANSWER:
        return "writer"
    if action == Action.REPLAN:
        return "planner"
    return "act"


# ------------------------------------------------------------------------------------------
# Act (tool execution)
# ------------------------------------------------------------------------------------------
def build_tool_args(action: Action, sq: str, query: str, top_k: int, explicit: dict[str, Any]) -> dict[str, Any]:
    if action == Action.SEARCH_DOCUMENTS:
        return {"query": str(explicit.get("query") or sq), "top_k": top_k}
    if action in (Action.LOOKUP_TABLE, Action.SEARCH_WEB):
        return {"query": str(explicit.get("query") or sq)}
    if action == Action.ANALYZE_VISUAL:
        args = {"question": str(explicit.get("question") or sq)}
        if explicit.get("figure_id"):
            args["figure_id"] = str(explicit["figure_id"])
        return args
    if action == Action.RUN_PYTHON:
        return {"code": str(explicit.get("code", "")), "purpose": query[:300]}
    return {}


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def act_node(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    decision = Decision.model_validate(state["next_action"])
    action = decision.action
    sq = current_subquestion(state)
    sq_index = len(state.get("resolved") or [])
    step = len(state.get("tool_history") or [])
    llm_calls: list[dict] = []
    args = build_tool_args(action, sq, state["query"], ctx.top_k, decision.args)

    if action == Action.RUN_PYTHON and not args["code"]:
        # Python Analyst: the LLM writes code over numeric evidence; the sandbox executes it.
        numeric = [e for e in (state.get("evidence") or []) if re.search(r"\d", e.content)]
        resp = ctx.llm.generate(python_messages(state["query"], numeric), max_tokens=256, purpose="python_analyst")
        llm_calls.append(_llm_record("python_analyst", resp))
        args["code"] = _extract_code(resp.text)

    # Working memory: never repeat an identical call; reformulate once, else skip.
    searched = set(state.get("searched") or [])
    if call_signature(ACTION_TOOL[action], args) in searched and "query" in args:
        args = {**args, "query": _reformulate(args["query"], state["query"])}
    signature = call_signature(ACTION_TOOL[action], args)
    if signature in searched:
        rec = ToolCallRecord(step=step, tool=ACTION_TOOL[action], arguments=args, ok=False,
                             error="duplicate call skipped (already in memory)", latency_ms=0.0,
                             timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        return {"tool_history": [rec], "llm_calls": llm_calls, "last_result": {"ok": False, "duplicate": True}}

    result, record = ctx.executor.call(ACTION_TOOL[action], args, step=step)
    tagged = [e.model_copy(update={"metadata": {**e.metadata, "subquestion": sq_index}}) for e in result.evidence]
    existing = state.get("evidence") or []
    new_count = len(merge_evidence(existing, tagged)) - len(existing)
    record = record.model_copy(update={"new_evidence": new_count})
    # For document searches also remember the un-reformulated signature for this sub-question.
    extra_sig = [call_signature("search_documents", {"query": sq})] if action == Action.SEARCH_DOCUMENTS else []
    return {
        "evidence": tagged, "tool_history": [record], "searched": [signature, *extra_sig],
        "llm_calls": llm_calls, "errors": [f"{record.tool}: {record.error}"] if record.error else [],
        "last_result": {"ok": result.ok, "tool": result.tool, "new_evidence": new_count,
                        "top_score": result.metadata.get("top_score")},
    }


# ------------------------------------------------------------------------------------------
# Critic
# ------------------------------------------------------------------------------------------
def heuristic_sufficiency(sq: str, evidence: list[Evidence], cfg: AgentConfig) -> tuple[bool, float]:
    texts = [e.content for e in evidence]
    cov = term_coverage(sq, texts)
    return (len(evidence) >= cfg.min_evidence and cov >= cfg.sufficient_score), cov


def critic_node(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    evidence = state.get("evidence") or []
    subs = state.get("subquestions") or [state["query"]]
    resolved = list(state.get("resolved") or [])
    sq_index = len(resolved) if len(resolved) < len(subs) else len(subs) - 1
    sq = subs[sq_index]
    sq_evidence = [e for e in evidence if e.metadata.get("subquestion") == sq_index] or evidence
    attempts = sum(1 for d in state.get("decisions") or [] if d.get("subquestion") == sq
                   and d.get("action") == Action.SEARCH_DOCUMENTS)
    llm_calls: list[dict] = []
    with get_tracer().span("agent.critic", kind="agent_node", mode=ctx.cfg.critic) as attrs:
        ok, cov = heuristic_sufficiency(sq, sq_evidence, ctx.cfg)
        missing = "" if ok else f"evidence for: {sq}"
        if ctx.cfg.critic == "llm" and ctx.caps.llm_can_reason and evidence:
            resp = ctx.llm.generate(critic_messages(sq, sq_evidence), max_tokens=120, purpose="critic")
            llm_calls.append(_llm_record("critic", resp))
            parsed = parse_json_object(resp.text)
            if parsed is not None and isinstance(parsed.get("sufficient"), bool):
                ok = parsed["sufficient"]
                missing = str(parsed.get("missing", ""))[:300]
        exhausted = attempts >= EXHAUSTED_ATTEMPTS
        if (ok or exhausted) and sq_index not in resolved and len(resolved) < len(subs):
            resolved = resolved + [sq_index]
        all_done = len(resolved) >= len(subs)
        # Multimodal gap: a table/figure question with no such evidence is not yet sufficient.
        cues = query_cues(state["query"])
        mods = {str(e.modality) for e in evidence}
        if cues["table"] and ctx.caps.tables and Modality.TABLE not in mods and not exhausted:
            all_done, missing = False, missing or "tabular evidence"
        attrs.update(sufficient=all_done, coverage=round(cov, 3), resolved=len(resolved))
    critique = {"subquestion": sq, "coverage": round(cov, 3), "sufficient": ok, "exhausted": exhausted,
                "missing": missing, "all_resolved": all_done}
    return {"resolved": resolved, "sufficient": all_done, "critiques": [critique], "llm_calls": llm_calls}


# ------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------
def select_evidence(evidence: list[Evidence], max_items: int = MAX_WRITER_EVIDENCE) -> list[Evidence]:
    """Pick a diverse evidence set: all computations/tables/figure readings first, then text
    round-robin across sub-questions (each list already in retrieval-rank order)."""
    special = [e for e in evidence if e.modality in (Modality.COMPUTATION, Modality.TABLE, Modality.FIGURE)]
    special = sorted(special, key=lambda e: e.modality != Modality.COMPUTATION)[: max_items // 2]
    groups: dict[Any, list[Evidence]] = {}
    for e in evidence:
        if e not in special:
            groups.setdefault(e.metadata.get("subquestion"), []).append(e)
    text: list[Evidence] = []
    while len(text) + len(special) < max_items and any(groups.values()):
        for g in groups.values():
            if g and len(text) + len(special) < max_items:
                text.append(g.pop(0))
    return special + text


def writer_node(state: ResearchState, ctx: AgentContext) -> dict[str, Any]:
    evidence = select_evidence(state.get("evidence") or [])
    status = state.get("status") if state.get("status") in ("budget_exhausted", "timeout") else "answered"
    with get_tracer().span("agent.writer", kind="agent_node", evidence=len(evidence)):
        if not evidence:
            answer = "I could not find evidence in the available sources to answer this question."
            return {"answer": answer, "citations": [], "status": status,
                    "report": {"answer": answer, "sources": [], "subquestions": state.get("subquestions", [])}}
        resp = ctx.llm.generate(grounded_messages(state["query"], evidence, WRITER_INSTRUCTIONS),
                                max_tokens=ctx.max_new_tokens,
                                purpose="writer")
    citations, invalid = resolve_citations(resp.text, evidence)
    cited = {c.marker for c in citations}
    report = {
        "question": state["query"],
        "plan": state.get("subquestions", []),
        "answer": resp.text,
        "sources": [{"marker": i, "label": e.citation_label(), "filename": e.filename, "page": e.page,
                     "url": e.url, "evidence_id": e.evidence_id, "modality": str(e.modality),
                     "tool": e.source_tool, "cited": i in cited, "excerpt": e.content[:300]}
                    for i, e in enumerate(evidence, start=1)],
        "invalid_citations": invalid,
        "tool_calls": len(state.get("tool_history") or []),
        "status": status,
    }
    return {"answer": resp.text, "citations": citations, "report": report, "status": status,
            "llm_calls": [_llm_record("writer", resp)]}
