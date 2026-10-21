"""The LangGraph research agent.

    START -> planner -> router --(tool action)--> act -> critic -> router ...
                          |--(replan)--> planner
                          `--(answer / budget)--> writer -> END

Why LangGraph (vs. a hand-written while-loop or a LangChain AgentExecutor)?
  * explicit, inspectable state machine with typed state and reducers
  * cycles with conditional edges (critic -> router -> act loops) as first-class
  * per-node streaming (`stream_mode="updates"`) that powers the API's SSE progress
  * checkpointing/interrupts available later (human-in-the-loop) without redesign
The node logic itself is plain Python in `atlas.agents.nodes` — LangGraph only
orchestrates, so the core behavior remains readable and unit-testable.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from atlas.agents.actions import ACTION_TOOL
from atlas.agents.nodes import (
    AgentContext,
    act_node,
    critic_node,
    planner_node,
    route_after_router,
    router_node,
    writer_node,
)
from atlas.agents.reducers import apply_update
from atlas.agents.state import ResearchState, initial_state
from atlas.common.ids import new_id
from atlas.common.types import Citation, Evidence
from atlas.observability.logging import get_logger
from atlas.observability.tracing import get_tracer
from atlas.tools.base import ToolCallRecord
from atlas.trajectory.schema import StepObservation, Trajectory, TrajectoryStep

log = get_logger("agent.graph")
_TOOL_ACTIONS = {a.value for a in ACTION_TOOL}  # actions that execute exactly one tool call


class ResearchResult(BaseModel):
    session_id: str
    trace_id: str | None = None
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)
    subquestions: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    tool_history: list[ToolCallRecord] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    critiques: list[dict[str, Any]] = Field(default_factory=list)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    status: str = ""
    iterations: int = 0
    latency_ms: float = 0.0
    policy: str = ""
    model: str = ""

    @property
    def prompt_tokens(self) -> int:
        return sum(c.get("prompt_tokens", 0) for c in self.llm_calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.get("completion_tokens", 0) for c in self.llm_calls)

    def to_trajectory(self) -> Trajectory:
        """Align router decisions with tool calls into (state, action, observation) steps."""
        steps: list[TrajectoryStep] = []
        tools = iter(self.tool_history)
        for i, d in enumerate(self.decisions):
            action = d["action"]
            rec = next(tools, None) if action in _TOOL_ACTIONS else None
            steps.append(TrajectoryStep(
                index=i, subquestion=d.get("subquestion", ""), action=action,
                tool=rec.tool if rec else None, arguments=rec.arguments if rec else d.get("args", {}),
                observation=StepObservation(ok=rec.ok, output=rec.output_preview, error=rec.error,
                                            new_evidence=rec.new_evidence) if rec else None,
                policy=d.get("policy", ""), rationale=d.get("rationale", ""), fallback=d.get("fallback", False),
                features=d.get("features"), log_prob=d.get("log_prob"), latency_ms=rec.latency_ms if rec else 0.0,
            ))
        return Trajectory(
            trajectory_id=new_id("traj_"), query=self.query, session_id=self.session_id, plan=self.subquestions,
            steps=steps, evidence=self.evidence, final_answer=self.answer, citations=self.citations,
            status=self.status, policy=self.policy, model=self.model, prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens, latency_ms=self.latency_ms,
            # writer evidence order == [n] citation markers; needed to rebuild the writer prompt for SFT
            metadata={"writer_evidence_ids": [s.get("evidence_id") for s in self.report.get("sources", [])]},
        )


class ResearchAgent:
    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        self.graph = self._build()

    def _build(self):
        g = StateGraph(ResearchState)
        g.add_node("planner", partial(planner_node, ctx=self.ctx))
        g.add_node("router", partial(router_node, ctx=self.ctx))
        g.add_node("act", partial(act_node, ctx=self.ctx))
        g.add_node("critic", partial(critic_node, ctx=self.ctx))
        g.add_node("writer", partial(writer_node, ctx=self.ctx))
        g.add_edge(START, "planner")
        g.add_edge("planner", "router")
        g.add_conditional_edges("router", route_after_router, {"act": "act", "planner": "planner", "writer": "writer"})
        g.add_edge("act", "critic")
        g.add_edge("critic", "router")
        g.add_edge("writer", END)
        return g.compile()

    @property
    def recursion_limit(self) -> int:
        # Each iteration visits router+act+critic (3 nodes); add headroom for planner/writer/replans.
        return 4 * self.ctx.cfg.max_iterations + 10

    def _initial(self, query: str, session_id: str | None) -> ResearchState:
        return initial_state(query, session_id or new_id("sess_"), time.time(), self.ctx.cfg.timeout_s)

    def stream(self, query: str, session_id: str | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield (node_name, partial_update) as the graph executes, then ('final', ResearchResult)."""
        state = dict(self._initial(query, session_id))
        t0 = time.perf_counter()
        with get_tracer().trace("research", query=query[:200], policy=self.ctx.policy.name) as trace_id:
            try:
                for chunk in self.graph.stream(state, config={"recursion_limit": self.recursion_limit},
                                               stream_mode="updates"):
                    for node, update in chunk.items():
                        state = apply_update(state, update or {})
                        yield node, update or {}
            except Exception as exc:
                log.exception("agent run failed")
                state = {**state, "status": "error", "errors": [*state.get("errors", []), str(exc)],
                         "answer": state.get("answer") or f"Research failed: {type(exc).__name__}: {exc}"}
        result = self._result(state, trace_id, (time.perf_counter() - t0) * 1000)
        yield "final", {"result": result}

    def run(self, query: str, session_id: str | None = None) -> ResearchResult:
        final = None
        for node, update in self.stream(query, session_id):
            if node == "final":
                final = update["result"]
        assert final is not None
        return final

    def _result(self, state: dict[str, Any], trace_id: str | None, latency_ms: float) -> ResearchResult:
        return ResearchResult(
            session_id=state["session_id"], trace_id=trace_id, query=state["query"], answer=state.get("answer", ""),
            citations=state.get("citations", []), report=state.get("report", {}),
            subquestions=state.get("subquestions", []), evidence=state.get("evidence", []),
            tool_history=state.get("tool_history", []), decisions=state.get("decisions", []),
            critiques=state.get("critiques", []), llm_calls=state.get("llm_calls", []),
            errors=state.get("errors", []), status=state.get("status", ""), iterations=state.get("iteration", 0),
            latency_ms=latency_ms, policy=self.ctx.policy.name, model=self.ctx.llm.model,
        )
