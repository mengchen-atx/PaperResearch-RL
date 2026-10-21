"""GRPO for the LLM router policy (TRL `GRPOTrainer` + LoRA).

The MLP router (atlas.rl.train) is cheap to train but only sees hand-built
features. An LLM router reads the full state text. To optimize it with RL we use
Group Relative Policy Optimization (Shao et al., 2024): for each prompt, sample a
group of G completions, score each, and use the group-normalized score as the
advantage — no value network needed, which suits LLM policies.

Formulation (one-step policy improvement with environment rollouts):
  prompt      router prompt at a state s_t taken from a collected trajectory
  completion  JSON action a_t
  reward      replay the trajectory prefix in ResearchEnv to reach s_t, execute
              a_t, then finish the episode with a fixed rollout policy (heuristic);
              reward = return from s_t onward. Unparseable/unavailable actions get
              `invalid_reward`.
This is a Monte-Carlo Q(s_t, a_t) estimate under the rollout policy — the classic
policy-iteration step — and every reward component is the same modular reward
used by the MLP trainer.

Compute: each reward needs an environment rollout, and GRPO needs G samples per
prompt. A meaningful run (hundreds of steps, 1.5B-7B policy) needs a CUDA GPU;
`--smoke` runs 1-2 optimizer steps on a 0.5B model to verify the pipeline.
"""

from __future__ import annotations

from typing import Any

from atlas.agents.actions import ACTIONS
from atlas.evaluation.benchmark import BenchmarkItem
from atlas.llm.json_utils import parse_json_object
from atlas.rl.env import ResearchEnv
from atlas.rl.train import heuristic_chooser
from atlas.training.sft_data import router_examples
from atlas.trajectory.schema import Trajectory

ACTION_NAMES = [a.value for a in ACTIONS]


def build_grpo_dataset(trajectories: list[Trajectory]) -> list[dict[str, Any]]:
    """One row per router decision; carries what the reward needs to replay the state."""
    rows = []
    for t in trajectories:
        bid = (t.metadata or {}).get("benchmark_id")
        if not bid:
            continue
        actions = [s.action for s in t.steps]
        for i, ex in enumerate(router_examples(t)):
            rows.append({"prompt": ex["prompt"], "benchmark_id": bid, "prefix_actions": actions[:i]})
    return rows


def completion_text(completion: Any) -> str:
    if isinstance(completion, list):  # conversational format: [{"role": "assistant", "content": ...}]
        return "".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def rollout_return(env: ResearchEnv, item: BenchmarkItem, prefix: list[str], action: str) -> float:
    env.reset(item)
    for a in prefix:
        out = env.step(ACTION_NAMES.index(a))
        if out.terminated or out.truncated:
            return 0.0  # prefix already ended the episode (should not happen for valid rows)
    mask = env.observe()[1]
    idx = ACTION_NAMES.index(action)
    if not mask[idx]:
        return env.reward_cfg.nu_invalid_action * -1.0
    out = env.step(idx)
    total = out.reward
    chooser = heuristic_chooser(env)
    while not (out.terminated or out.truncated):
        obs, mask = env.observe()
        out = env.step(chooser(obs, mask))
        total += out.reward
    return total


def make_reward_fn(env: ResearchEnv, items: dict[str, BenchmarkItem], invalid_reward: float = -0.5):
    def reward_fn(prompts, completions, benchmark_id, prefix_actions, **_: Any) -> list[float]:
        rewards = []
        for comp, bid, prefix in zip(completions, benchmark_id, prefix_actions, strict=True):
            parsed = parse_json_object(completion_text(comp)) or {}
            action = str(parsed.get("action", "")).strip().lower()
            if action not in ACTION_NAMES or bid not in items:
                rewards.append(invalid_reward)
                continue
            rewards.append(rollout_return(env, items[bid], list(prefix), action))
        return rewards

    reward_fn.__name__ = "env_rollout_return"
    return reward_fn
