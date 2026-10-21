"""State featurization + action masking.

The same fixed-length feature vector is used by
  * the learned (RL) router policy at inference time, and
  * the RL environment's observation,
so a policy trained in `atlas.rl` runs unchanged inside the production agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atlas.agents.actions import ACTIONS, Action, has_numbers, query_cues, term_coverage
from atlas.agents.state import ResearchState
from atlas.common.types import Modality

FEATURE_DIM = 19 + 2 * len(ACTIONS)
MAX_REPLANS = 1


@dataclass(frozen=True)
class Capabilities:
    """What the current deployment can actually do; drives the action mask."""

    python: bool = False
    visual: bool = False
    web: bool = False
    tables: bool = False
    llm_can_reason: bool = False  # False for the extractive provider (cannot write code/plans)


def current_subquestion(state: ResearchState) -> str:
    subs = state.get("subquestions") or [state["query"]]
    resolved = set(state.get("resolved") or [])
    for i, sq in enumerate(subs):
        if i not in resolved:
            return sq
    return subs[-1]


def action_counts(state: ResearchState) -> dict[Action, int]:
    counts = {a: 0 for a in ACTIONS}
    for d in state.get("decisions") or []:
        counts[Action(d["action"])] += 1
    return counts


def featurize(state: ResearchState, max_iterations: int, max_tool_calls: int) -> np.ndarray:
    query = state["query"]
    cues = query_cues(query)
    evidence = state.get("evidence") or []
    subs = state.get("subquestions") or [query]
    resolved = state.get("resolved") or []
    texts = [e.content for e in evidence]
    mods = {str(e.modality) for e in evidence}
    history = state.get("tool_history") or []
    last = history[-1] if history else None
    counts = action_counts(state)
    decisions = state.get("decisions") or []
    last_action = Action(decisions[-1]["action"]) if decisions else None

    f = [
        float(cues["table"]), float(cues["figure"]), float(cues["compute"]), float(cues["compare"]),
        min(len(subs), 3) / 3,
        len(resolved) / max(len(subs), 1),
        min(state.get("iteration", 0) / max_iterations, 1.0),
        min(len(history) / max_tool_calls, 1.0),
        min(len(evidence) / 10, 1.0),
        float(Modality.TEXT in mods), float(Modality.TABLE in mods), float(Modality.FIGURE in mods),
        float(Modality.COMPUTATION in mods),
        term_coverage(current_subquestion(state), texts),
        term_coverage(query, texts),
        float(bool(last and last.new_evidence > 0)),
        float(bool(last and last.ok)),
        float(any(has_numbers(t) for t in texts)),
        1.0,  # bias
    ]
    f += [min(counts[a], 3) / 3 for a in ACTIONS]
    f += [float(last_action == a) for a in ACTIONS]
    return np.asarray(f, dtype=np.float32)


def action_mask(state: ResearchState, caps: Capabilities) -> list[bool]:
    evidence = state.get("evidence") or []
    numeric = any(has_numbers(e.content) for e in evidence)
    allowed = {
        Action.SEARCH_DOCUMENTS: True,
        Action.LOOKUP_TABLE: caps.tables,
        Action.ANALYZE_VISUAL: caps.visual,
        Action.RUN_PYTHON: caps.python and caps.llm_can_reason and numeric,
        Action.SEARCH_WEB: caps.web,
        Action.REPLAN: (state.get("replans", 0) < MAX_REPLANS),
        Action.ANSWER: True,
    }
    return [allowed[a] for a in ACTIONS]
