"""Apply a node's partial update to a state dict, mirroring the LangGraph reducers
declared in `ResearchState`. Used by `ResearchAgent.stream` (to assemble the final
result) and by the RL environment (which drives the nodes without LangGraph)."""

from __future__ import annotations

from typing import Any

from atlas.memory.session import merge_evidence

ACCUMULATED = frozenset({"tool_history", "decisions", "critiques", "searched", "llm_calls", "errors"})


def apply_update(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(state)
    for key, value in update.items():
        if key in ACCUMULATED:
            out[key] = [*out.get(key, []), *value]
        elif key == "evidence":
            out[key] = merge_evidence(out.get(key, []), value)
        else:
            out[key] = value
    return out
