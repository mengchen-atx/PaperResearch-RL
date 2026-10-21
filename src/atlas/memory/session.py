"""Working memory for one research session (pure functions over agent state).

The agent state *is* the working memory; these helpers answer the questions the
agent needs to avoid wasted work:
  * Has this exact tool call already been made?      -> call_signature / already_done
  * Is this evidence already in the pool?            -> merge_evidence (dedupe reducer)
  * Which sub-questions are still unresolved?        -> unresolved_subquestions
"""

from __future__ import annotations

import json
from typing import Any

from atlas.common.types import Evidence


def call_signature(tool: str, args: dict[str, Any]) -> str:
    """Normalized identity of a tool call (case/whitespace-insensitive for text args)."""
    norm = {k: " ".join(v.lower().split()) if isinstance(v, str) else v for k, v in sorted(args.items())}
    return f"{tool}:{json.dumps(norm, sort_keys=True, default=str)}"


def already_done(searched: list[str], tool: str, args: dict[str, Any]) -> bool:
    return call_signature(tool, args) in set(searched)


def _content_key(ev: Evidence) -> str:
    return " ".join(ev.content.lower().split())[:300]


def merge_evidence(existing: list[Evidence], new: list[Evidence]) -> list[Evidence]:
    """LangGraph reducer: append only evidence not already present (by id or content)."""
    ids = {e.evidence_id for e in existing}
    contents = {_content_key(e) for e in existing}
    out = list(existing)
    for ev in new:
        key = _content_key(ev)
        if ev.evidence_id in ids or key in contents:
            continue
        ids.add(ev.evidence_id)
        contents.add(key)
        out.append(ev)
    return out


def unresolved_subquestions(subquestions: list[str], resolved: list[int]) -> list[int]:
    done = set(resolved)
    return [i for i in range(len(subquestions)) if i not in done]
