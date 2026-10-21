"""LangGraph state for the research agent.

Fields annotated with a reducer are *accumulated* across nodes (append/merge);
all others are overwritten by the node that returns them. Nodes never mutate the
incoming state — they return a partial update dict, which LangGraph applies.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from atlas.common.types import Citation, Evidence
from atlas.memory.session import merge_evidence
from atlas.tools.base import ToolCallRecord


class ResearchState(TypedDict, total=False):
    # --- input & plan -------------------------------------------------------------
    query: str
    session_id: str
    subquestions: list[str]
    resolved: list[int]  # indices of resolved subquestions
    plan_rationale: str
    replans: int
    # --- research progress (working memory) ------------------------------------
    evidence: Annotated[list[Evidence], merge_evidence]
    tool_history: Annotated[list[ToolCallRecord], operator.add]
    decisions: Annotated[list[dict[str, Any]], operator.add]
    critiques: Annotated[list[dict[str, Any]], operator.add]
    searched: Annotated[list[str], operator.add]
    llm_calls: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    iteration: int
    next_action: dict[str, Any] | None
    last_result: dict[str, Any] | None
    sufficient: bool
    # --- budget -------------------------------------------------------------------
    started_at: float
    deadline: float
    # --- output -------------------------------------------------------------------
    answer: str
    citations: list[Citation]
    report: dict[str, Any]
    status: str  # running | answered | budget_exhausted | timeout | error


def initial_state(query: str, session_id: str, started_at: float, timeout_s: float) -> ResearchState:
    return ResearchState(
        query=query, session_id=session_id, subquestions=[], resolved=[], plan_rationale="", replans=0,
        evidence=[], tool_history=[], decisions=[], critiques=[], searched=[], llm_calls=[], errors=[],
        iteration=0, next_action=None, last_result=None, sufficient=False, started_at=started_at,
        deadline=started_at + timeout_s, answer="", citations=[], report={}, status="running",
    )
