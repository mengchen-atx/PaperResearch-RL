"""Router policy network: features -> masked action distribution (+ value estimate).

A small MLP actor-critic. The actor outputs logits over the 7 actions; invalid
actions (per the action mask) get -inf logits so they are never sampled. The
critic head estimates V(s), used as a variance-reducing baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from atlas.agents.actions import ACTIONS
from atlas.agents.features import FEATURE_DIM


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int = FEATURE_DIM, n_actions: int = len(ACTIONS), hidden: int = 64) -> None:
        super().__init__()
        self.obs_dim, self.n_actions, self.hidden = obs_dim, n_actions, hidden
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        h = self.body(obs)
        logits = self.actor(h).masked_fill(~mask, float("-inf"))
        return torch.distributions.Categorical(logits=logits), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False) -> tuple[int, float]:
        dist, _ = self(torch.as_tensor(obs).unsqueeze(0), torch.as_tensor(mask).unsqueeze(0))
        a = int(dist.probs.argmax(-1).item()) if greedy else int(dist.sample().item())
        return a, float(dist.log_prob(torch.tensor([a])).item())

    def save(self, path: Path, extra: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                    "hidden": self.hidden, "actions": [a.value for a in ACTIONS], "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: Path) -> PolicyNetwork:
        if not Path(path).exists():
            from atlas.common.errors import ConfigError

            raise ConfigError(f"No learned policy at {path}. Train one with `python scripts/train_rl.py`.")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt["actions"] != [a.value for a in ACTIONS]:
            raise ValueError("Checkpoint action space does not match the current Action enum")
        net = cls(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden"])
        net.load_state_dict(ckpt["state_dict"])
        return net.eval()
