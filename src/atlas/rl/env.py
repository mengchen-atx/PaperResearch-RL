"""Research-agent MDP environment.

Formalization
-------------
  state s_t        ResearchState: query, plan, evidence pool, tool history,
                   critic verdicts, iteration/budget counters
  observation o_t  featurize(s_t) in R^FEATURE_DIM (+ action mask)
  action a_t       one of Action (search_documents, lookup_table, analyze_visual,
                   run_python, search_web, replan, answer)
  transition       exactly the production agent's nodes:
                   tool actions -> act_node + critic_node; replan -> planner_node;
                   answer -> writer_node (terminal)
  reward r_t       atlas.rl.reward (terminal success/citations + shaped evidence
                   quality - tool cost - step penalty)
  termination      `answer` action; truncation when max_iterations/max_tool_calls
                   is reached (forced writer, as in production)

Because the env drives the *same* node functions and tools as the LangGraph
agent, a policy trained here transfers to `LearnedPolicy` without adaptation.

Efficiency: tools are deterministic given their arguments, so results are
memoized (`CachingExecutor`). After the first pass over the training questions
most steps are cache hits, which makes thousands of episodes feasible on CPU.
The terminal writer is pluggable; the default ExtractiveLLM keeps episodes
cheap (see docs/design/agentic_rl.md for the trade-off).
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from atlas.agents.actions import ACTIONS, Action, Decision
from atlas.agents.features import Capabilities, action_mask, current_subquestion, featurize
from atlas.agents.nodes import AgentContext, act_node, critic_node, planner_node, writer_node
from atlas.agents.policies import HeuristicPolicy
from atlas.agents.reducers import apply_update
from atlas.agents.state import ResearchState, initial_state
from atlas.common.config import AgentConfig
from atlas.evaluation.benchmark import BenchmarkItem
from atlas.llm.base import LLMClient
from atlas.rl.reward import RewardConfig, evidence_potential, step_reward, terminal_reward
from atlas.tools.base import ToolCallRecord, ToolExecutor, ToolResult


class CachingExecutor(ToolExecutor):
    """Memoizes tool results by (tool, args). Only valid for deterministic tools."""

    def __init__(self, inner: ToolExecutor) -> None:
        self.inner = inner
        self.tools = inner.tools
        self.timeout_s = inner.timeout_s
        self._cache: dict[str, ToolResult] = {}
        self.hits = 0
        self.misses = 0

    def call(self, name: str, arguments: dict[str, Any], step: int = 0) -> tuple[ToolResult, ToolCallRecord]:
        import json

        key = f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"
        if key in self._cache:
            self.hits += 1
            res = self._cache[key]
            rec = ToolCallRecord(step=step, tool=name, arguments=arguments, ok=res.ok, error=res.error,
                                 latency_ms=res.latency_ms, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 num_evidence=len(res.evidence), output_preview=res.output[:300])
            return res, rec
        self.misses += 1
        res, rec = self.inner.call(name, arguments, step)
        self._cache[key] = res
        return res, rec


@dataclass
class StepOutput:
    obs: np.ndarray
    mask: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


class ResearchEnv:
    def __init__(self, executor: ToolExecutor, caps: Capabilities, agent_cfg: AgentConfig, answerer: LLMClient,
                 reward_cfg: RewardConfig | None = None, top_k: int = 5) -> None:
        # Deterministic, cheap components inside RL rollouts: heuristic planner & critic.
        cfg = agent_cfg.model_copy(update={"planner": "heuristic", "critic": "heuristic"})
        if answerer.provider == "extractive":
            # The extractive answerer cannot write code, so the Python action is unavailable.
            caps = dataclasses.replace(caps, llm_can_reason=False)
        self.cfg = cfg
        self.reward_cfg = reward_cfg or RewardConfig()
        self.executor = executor if isinstance(executor, CachingExecutor) else CachingExecutor(executor)
        self.ctx = AgentContext(cfg=cfg, llm=answerer, executor=self.executor, policy=HeuristicPolicy(),
                                caps=caps, top_k=top_k, max_new_tokens=256)
        self.state: ResearchState | None = None
        self.item: BenchmarkItem | None = None
        self.phi = 0.0

    # --- gym-style API ----------------------------------------------------------------
    def reset(self, item: BenchmarkItem) -> tuple[np.ndarray, np.ndarray]:
        self.item = item
        state = initial_state(item.question, f"rl-{item.id}", time.time(), timeout_s=1e9)
        self.state = apply_update(state, planner_node(state, self.ctx))
        self.phi = 0.0
        return self.observe()

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.state is not None
        return (featurize(self.state, self.cfg.max_iterations, self.cfg.max_tool_calls),
                np.asarray(action_mask(self.state, self.ctx.caps), dtype=bool))

    def step(self, action_idx: int) -> StepOutput:
        assert self.state is not None and self.item is not None, "call reset() first"
        state = self.state
        action = ACTIONS[action_idx]
        mask = action_mask(state, self.ctx.caps)
        invalid = not mask[action_idx]
        if invalid:  # masked action chosen: treat as a no-op wasted step
            action = Action.SEARCH_DOCUMENTS if not state.get("evidence") else Action.ANSWER
        decision = Decision(action=action, policy="rl")
        state = apply_update(state, {
            "next_action": decision.model_dump(mode="json"),
            "decisions": [{**decision.model_dump(mode="json"), "iteration": state.get("iteration", 0),
                           "subquestion": current_subquestion(state)}],
            "iteration": state.get("iteration", 0) + 1,
        })
        components: dict[str, float] = {}
        terminated = truncated = False
        if action == Action.ANSWER:
            state, term = self._finish(state)
            components.update(term)
            terminated = True
        else:
            if action == Action.REPLAN:
                state = apply_update(state, {"replans": state.get("replans", 0) + 1})
                state = apply_update(state, planner_node(state, self.ctx))
            else:
                state = apply_update(state, act_node(state, self.ctx))
                state = apply_update(state, critic_node(state, self.ctx))
            if (state["iteration"] >= self.cfg.max_iterations
                    or len(state.get("tool_history") or []) >= self.cfg.max_tool_calls):
                state = apply_update(state, {"status": "budget_exhausted"})
                state, term = self._finish(state)
                components.update(term)
                truncated = True
        phi_after = evidence_potential(state.get("evidence") or [], self.item)
        components.update(step_reward(self.reward_cfg, action, self.phi, phi_after, invalid))
        self.phi = phi_after
        self.state = state
        obs, next_mask = self.observe()
        return StepOutput(obs, next_mask, float(sum(components.values())), terminated, truncated,
                          {"reward_components": components, "action": action.value,
                           "answer": state.get("answer", "")})

    def _finish(self, state: ResearchState) -> tuple[ResearchState, dict[str, float]]:
        state = apply_update(state, writer_node(state, self.ctx))
        locs = [(s.get("filename"), s.get("page")) for s in state.get("report", {}).get("sources", [])]
        return state, terminal_reward(self.reward_cfg, state.get("answer", ""), self.item, locs)
