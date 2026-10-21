"""Train the agent's router policy with policy-gradient RL in the research environment.

    python scripts/train_rl.py [-c configs/rl/reinforce.yaml] [--iterations N] [--smoke]

Writes data/processed/rl/{policy.pt,results.json}. Use the policy in the agent with
`atlas research "..." --policy learned` or configs/experiments/agent_learned.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from atlas.common.config import load_config
from atlas.evaluation.benchmark import load_benchmark, split_items
from atlas.llm.providers import ExtractiveLLM
from atlas.observability.logging import configure_logging
from atlas.rl.env import ResearchEnv
from atlas.rl.reward import RewardConfig
from atlas.rl.train import RLTrainConfig, train
from atlas.runtime import AtlasRuntime


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/rl/reinforce.yaml")
    ap.add_argument("--benchmark", default="data/benchmarks/atlas_papers_v1.jsonl")
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--smoke", action="store_true", help="2 iterations x 2 episodes, for CI")
    ap.add_argument("--no-vlm", action="store_true", help="disable the visual action (faster rollouts)")
    args = ap.parse_args()
    rl_cfg = yaml.safe_load(Path(args.config).read_text())
    cfg = load_config()
    if args.no_vlm:
        cfg = cfg.with_overrides({"vlm": {"provider": "none"}})
    configure_logging(cfg.observability.log_level)
    train_cfg = RLTrainConfig(**rl_cfg.get("train", {}))
    if args.iterations:
        train_cfg = train_cfg.model_copy(update={"iterations": args.iterations})
    if args.smoke:
        train_cfg = train_cfg.model_copy(update={"iterations": 2, "episodes_per_update": 2})
    items = load_benchmark(Path(args.benchmark))
    split = rl_cfg.get("split", {})
    train_items, test_items = split_items(items, split.get("test_fraction", 0.3), split.get("seed", 0))
    if args.smoke:
        train_items, test_items = train_items[:3], test_items[:2]

    rt = AtlasRuntime(cfg)
    env = ResearchEnv(rt.executor, rt.capabilities(), cfg.agent, ExtractiveLLM(),
                      RewardConfig(**rl_cfg.get("reward", {})), top_k=cfg.retrieval.top_k)
    out_dir = cfg.resolve_path(rl_cfg.get("out_dir", "data/processed/rl") + ("_smoke" if args.smoke else ""))
    results = train(env, train_items, test_items, train_cfg, out_dir)
    print(json.dumps({k: {m: (round(v, 3) if isinstance(v, float) else v) for m, v in r.items()
                          if m != "reward_components"} for k, r in results["test"].items()}, indent=1))
    print(f"policy -> {out_dir / 'policy.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
