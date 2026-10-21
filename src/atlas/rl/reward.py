"""Modular reward for the research-agent MDP.

Episode return (undiscounted, gamma_discount=1 by default):

    R = alpha * task_success            (terminal: answer matches gold)
      + beta  * citation_quality        (terminal: cited evidence is gold-relevant)
      + gamma * evidence_quality        (shaped: potential-based, see below)
      - lambda * sum_t tool_cost(a_t)   (per step: relative tool cost)
      - mu     * sum_t step_penalty     (per step: discourages long trajectories)
      - nu     * invalid/duplicate actions

Evidence quality uses potential-based shaping (Ng et al., 1999):
    F(s, s') = Phi(s') - Phi(s),  Phi(s) = recall of gold evidence units in the pool
Shaping of this form densifies the signal without changing the optimal policy,
and it telescopes to gamma * Phi(s_T) over an episode.

Every component is computed separately and returned in `info["reward_components"]`
so experiments can inspect *why* a policy is rewarded.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from atlas.agents.actions import Action
from atlas.common.types import Evidence
from atlas.evaluation import metrics as M
from atlas.evaluation.benchmark import BenchmarkItem


class RewardConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha_task_success: float = 1.0
    beta_citation_quality: float = 0.3
    gamma_evidence_quality: float = 0.5
    lambda_tool_cost: float = 0.03
    mu_step_penalty: float = 0.02
    nu_invalid_action: float = 0.1
    tool_costs: dict[str, float] = {
        Action.SEARCH_DOCUMENTS.value: 1.0,
        Action.LOOKUP_TABLE.value: 1.0,
        Action.ANALYZE_VISUAL.value: 3.0,
        Action.RUN_PYTHON.value: 1.0,
        Action.SEARCH_WEB.value: 2.0,
        Action.REPLAN.value: 0.5,
        Action.ANSWER.value: 0.0,
    }


def evidence_potential(evidence: list[Evidence], item: BenchmarkItem) -> float:
    locs = [(e.filename, e.page) for e in evidence]
    return M.recall_at_k(locs, item.evidence, max(len(locs), 1)) if locs else 0.0


def step_reward(cfg: RewardConfig, action: Action, phi_before: float, phi_after: float,
                invalid: bool) -> dict[str, float]:
    return {
        "evidence_shaping": cfg.gamma_evidence_quality * (phi_after - phi_before),
        "tool_cost": -cfg.lambda_tool_cost * cfg.tool_costs.get(action.value, 1.0),
        "step_penalty": -cfg.mu_step_penalty,
        "invalid": -cfg.nu_invalid_action if invalid else 0.0,
    }


def terminal_reward(cfg: RewardConfig, answer: str, item: BenchmarkItem,
                    cited_locs: list[M.Location]) -> dict[str, float]:
    success = float(M.answer_correct(answer, item))
    cit = M.citation_metrics(answer, cited_locs, item.evidence)["citation_precision"] or 0.0
    return {"task_success": cfg.alpha_task_success * success, "citation_quality": cfg.beta_citation_quality * cit}
