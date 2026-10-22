"""Router policies: decide the next action from the research state.

  * HeuristicPolicy — hand-written rules keyed on query cues and progress. A
    transparent baseline (and the fallback for invalid LLM output). NOT learned.
  * LLMPolicy       — the LLM reads the state and emits a JSON action (ReAct-style
    routing). Invalid/unavailable actions fall back to the heuristic and the
    fallback is recorded (a measurable agent metric).
  * SingleSearchPolicy — fixed "search once, then answer" pipeline. A control for
    the learned router, which converges to this behaviour on the v1 benchmark.
  * LearnedPolicy   — an MLP over `featurize(state)` trained with policy-gradient
    RL in `atlas.rl`. Loaded from a checkpoint; greedy at inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from atlas.agents.actions import ACTIONS, Action, Decision, query_cues
from atlas.agents.features import Capabilities, action_counts, action_mask, current_subquestion, featurize
from atlas.agents.prompts import router_messages
from atlas.agents.state import ResearchState
from atlas.llm.base import LLMClient
from atlas.llm.json_utils import parse_json_object
from atlas.memory.session import call_signature


class Policy(Protocol):
    name: str

    def decide(self, state: ResearchState, caps: Capabilities) -> Decision: ...


class HeuristicPolicy:
    name = "heuristic"

    def decide(self, state: ResearchState, caps: Capabilities) -> Decision:
        mask = dict(zip(ACTIONS, action_mask(state, caps), strict=True))
        counts = action_counts(state)
        cues = query_cues(state["query"])
        sq = current_subquestion(state)
        searched = set(state.get("searched") or [])

        def pick(action: Action, why: str) -> Decision:
            return Decision(action=action, rationale=why, policy=self.name)

        if call_signature("search_documents", {"query": sq}) not in searched:
            return pick(Action.SEARCH_DOCUMENTS, "current sub-question not searched yet")
        if cues["table"] and mask[Action.LOOKUP_TABLE] and counts[Action.LOOKUP_TABLE] == 0:
            return pick(Action.LOOKUP_TABLE, "question asks for tabular/quantitative results")
        if cues["figure"] and mask[Action.ANALYZE_VISUAL] and counts[Action.ANALYZE_VISUAL] == 0:
            return pick(Action.ANALYZE_VISUAL, "question refers to a figure/chart")
        if cues["compute"] and mask[Action.RUN_PYTHON] and counts[Action.RUN_PYTHON] == 0:
            return pick(Action.RUN_PYTHON, "question requires arithmetic over retrieved numbers")
        if not state.get("sufficient"):
            if mask[Action.REPLAN]:
                return pick(Action.REPLAN, "critic judged evidence insufficient; re-plan")
            if mask[Action.SEARCH_WEB] and counts[Action.SEARCH_WEB] == 0:
                return pick(Action.SEARCH_WEB, "local corpus insufficient; try the web")
        return pick(Action.ANSWER, "evidence judged sufficient or options exhausted")


class SingleSearchPolicy:
    name = "single_search"

    def decide(self, state: ResearchState, caps: Capabilities) -> Decision:
        if action_counts(state)[Action.SEARCH_DOCUMENTS] == 0:
            return Decision(action=Action.SEARCH_DOCUMENTS, rationale="fixed baseline: one search",
                            policy=self.name)
        return Decision(action=Action.ANSWER, rationale="fixed baseline: answer after one search", policy=self.name)


class LLMPolicy:
    name = "llm"

    def __init__(self, llm: LLMClient, fallback: Policy | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or HeuristicPolicy()

    def decide(self, state: ResearchState, caps: Capabilities) -> Decision:
        mask = action_mask(state, caps)
        available = [a.value for a, ok in zip(ACTIONS, mask, strict=True) if ok]
        history = [d["action"] for d in state.get("decisions") or []]
        msgs = router_messages(state["query"], current_subquestion(state), state.get("evidence") or [],
                               history, available)
        resp = self.llm.generate(msgs, max_tokens=200, purpose="router")
        parsed = parse_json_object(resp.text) or {}
        action = str(parsed.get("action", "")).strip().lower()
        args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}
        usage = {"model": resp.model, "prompt_tokens": resp.usage.prompt_tokens,
                 "completion_tokens": resp.usage.completion_tokens, "latency_ms": round(resp.latency_ms, 1),
                 "cached": resp.cached}
        if action in available:
            return Decision(action=Action(action), args=args, rationale=str(parsed.get("reason", ""))[:300],
                            policy=self.name, llm_call=usage)
        fb = self.fallback.decide(state, caps)
        return fb.model_copy(update={"policy": self.name, "fallback": True, "llm_call": usage,
                                     "rationale": f"invalid LLM action {action!r}; fallback: {fb.rationale}"})


class LearnedPolicy:
    """Wraps an RL-trained `PolicyNetwork` checkpoint (see atlas.rl.policy)."""

    name = "learned"

    def __init__(self, checkpoint: Path, max_iterations: int, max_tool_calls: int, greedy: bool = True) -> None:
        from atlas.rl.policy import PolicyNetwork

        self.net = PolicyNetwork.load(checkpoint)
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.greedy = greedy

    def decide(self, state: ResearchState, caps: Capabilities) -> Decision:
        feats = featurize(state, self.max_iterations, self.max_tool_calls)
        mask = np.asarray(action_mask(state, caps), dtype=bool)
        idx, logp = self.net.act(feats, mask, greedy=self.greedy)
        return Decision(action=ACTIONS[idx], policy=self.name, features=feats.tolist(), log_prob=logp,
                        rationale="learned policy (argmax over masked action logits)")
