"""Policy-gradient training of the router policy (REINFORCE with a learned baseline).

Algorithm (Williams 1992 + value baseline, i.e. Monte-Carlo advantage actor-critic):
    for each iteration:
        collect `episodes_per_update` episodes with the stochastic policy
        G_t   = sum_{k>=t} discount^(k-t) r_k            (reward-to-go)
        A_t   = G_t - V(s_t)                               (advantage)
        loss  = -E[log pi(a_t|s_t) * A_t]                  (actor)
              + value_coef * E[(G_t - V(s_t))^2]           (critic)
              - entropy_coef * E[H(pi(.|s_t))]             (exploration)
Why this and not PPO? Episodes are short (<= max_iterations steps) and the action
space is small, so vanilla policy gradient with a baseline is stable and simple
to read. PPO's clipped objective becomes worthwhile with larger policies (e.g.
the LLM policy, see `atlas.rl.grpo`).

Evaluation compares greedy learned policy vs HeuristicPolicy vs uniform-random
policy on held-out questions in the *same* environment, reporting mean return,
task success, tool calls and each reward component.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel

from atlas.agents.actions import ACTIONS
from atlas.agents.policies import HeuristicPolicy
from atlas.evaluation.benchmark import BenchmarkItem
from atlas.observability.logging import get_logger
from atlas.rl.env import ResearchEnv
from atlas.rl.policy import PolicyNetwork

log = get_logger("rl.train")


class RLTrainConfig(BaseModel):
    iterations: int = 60
    episodes_per_update: int = 16
    lr: float = 3e-3
    discount: float = 1.0
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    hidden: int = 64
    seed: int = 0
    grad_clip: float = 1.0


@dataclass
class Episode:
    obs: list[np.ndarray]
    masks: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    info: dict[str, Any]


def run_episode(env: ResearchEnv, item: BenchmarkItem, choose) -> Episode:
    obs, mask = env.reset(item)
    ep = Episode([], [], [], [], {})
    components: dict[str, float] = {}
    done = False
    while not done:
        a = choose(obs, mask)
        out = env.step(a)
        ep.obs.append(obs)
        ep.masks.append(mask)
        ep.actions.append(a)
        ep.rewards.append(out.reward)
        for k, v in out.info["reward_components"].items():
            components[k] = components.get(k, 0.0) + v
        obs, mask = out.obs, out.mask
        done = out.terminated or out.truncated
    state = env.state or {}
    ep.info = {"components": components, "answer": out.info.get("answer", ""),
               "tool_calls": len(state.get("tool_history") or []),
               "success": components.get("task_success", 0.0) > 0,
               "actions": [ACTIONS[a].value for a in ep.actions]}
    return ep


def reward_to_go(rewards: list[float], discount: float) -> list[float]:
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + discount * g
        out.append(g)
    return out[::-1]


def evaluate_policy(env: ResearchEnv, items: list[BenchmarkItem], choose) -> dict[str, Any]:
    eps = [run_episode(env, it, choose) for it in items]
    comps: dict[str, float] = {}
    for e in eps:
        for k, v in e.info["components"].items():
            comps[k] = comps.get(k, 0.0) + v / len(eps)
    action_hist: dict[str, int] = {}
    for e in eps:
        for a in e.info["actions"]:
            action_hist[a] = action_hist.get(a, 0) + 1
    return {"n": len(eps), "mean_return": float(np.mean([sum(e.rewards) for e in eps])),
            "success_rate": float(np.mean([e.info["success"] for e in eps])),
            "mean_tool_calls": float(np.mean([e.info["tool_calls"] for e in eps])),
            "mean_steps": float(np.mean([len(e.actions) for e in eps])),
            "reward_components": comps, "action_histogram": action_hist}


def heuristic_chooser(env: ResearchEnv):
    policy = HeuristicPolicy()

    def choose(obs, mask):
        d = policy.decide(env.state, env.ctx.caps)
        return ACTIONS.index(d.action)

    return choose


def random_chooser(seed: int = 0):
    rng = np.random.default_rng(seed)

    def choose(obs, mask):
        return int(rng.choice(np.flatnonzero(mask)))

    return choose


def train(env: ResearchEnv, train_items: list[BenchmarkItem], test_items: list[BenchmarkItem],
          cfg: RLTrainConfig, out_dir: Path) -> dict[str, Any]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    net = PolicyNetwork(hidden=cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    history = []
    t0 = time.perf_counter()

    def sample(obs, mask):
        return net.act(obs, mask, greedy=False)[0]

    for it in range(cfg.iterations):
        batch = [run_episode(env, random.choice(train_items), sample) for _ in range(cfg.episodes_per_update)]
        obs = torch.as_tensor(np.stack([o for e in batch for o in e.obs]))
        masks = torch.as_tensor(np.stack([m for e in batch for m in e.masks]))
        acts = torch.as_tensor([a for e in batch for a in e.actions])
        rtg = torch.as_tensor([g for e in batch for g in reward_to_go(e.rewards, cfg.discount)], dtype=torch.float32)
        dist, values = net(obs, masks)
        adv = rtg - values.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
        loss_pi = -(dist.log_prob(acts) * adv).mean()
        loss_v = ((rtg - values) ** 2).mean()
        entropy = dist.entropy().mean()
        loss = loss_pi + cfg.value_coef * loss_v - cfg.entropy_coef * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        rec = {"iteration": it, "mean_return": float(np.mean([sum(e.rewards) for e in batch])),
               "success_rate": float(np.mean([e.info["success"] for e in batch])),
               "mean_tool_calls": float(np.mean([e.info["tool_calls"] for e in batch])),
               "entropy": float(entropy.item()), "loss": float(loss.item()),
               "cache_hit_rate": env.executor.hits / max(env.executor.hits + env.executor.misses, 1)}
        history.append(rec)
        if it % 5 == 0 or it == cfg.iterations - 1:
            log.info("rl iteration", extra={k: round(v, 3) if isinstance(v, float) else v for k, v in rec.items()})

    def greedy(obs, mask):
        return net.act(obs, mask, greedy=True)[0]

    results = {
        "config": cfg.model_dump(), "train_questions": len(train_items), "test_questions": len(test_items),
        "train_seconds": time.perf_counter() - t0, "reward_config": env.reward_cfg.model_dump(),
        "test": {"learned_greedy": evaluate_policy(env, test_items, greedy),
                 "heuristic": evaluate_policy(env, test_items, heuristic_chooser(env)),
                 "random": evaluate_policy(env, test_items, random_chooser(cfg.seed))},
        "train_eval": {"learned_greedy": evaluate_policy(env, train_items, greedy),
                       "heuristic": evaluate_policy(env, train_items, heuristic_chooser(env))},
        "history": history,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    net.save(out_dir / "policy.pt", extra={"trained_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results
