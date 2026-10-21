"""GRPO training of the LLM router (TRL GRPOTrainer + LoRA), rewards from ResearchEnv rollouts.

    python scripts/train_grpo.py --smoke                       # 2 steps, 0.5B, verifies the pipeline
    python scripts/train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --max-steps 300   # needs a CUDA GPU

Trajectories must carry benchmark ids (collected via scripts/run_agent_benchmark.py).
Optionally start from the SFT adapter (--init-adapter data/processed/sft/adapter).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atlas.common.config import load_config
from atlas.common.device import resolve_device
from atlas.evaluation.benchmark import load_benchmark, split_items
from atlas.llm.providers import ExtractiveLLM
from atlas.observability.logging import configure_logging
from atlas.rl.env import ResearchEnv
from atlas.rl.grpo import build_grpo_dataset, make_reward_fn
from atlas.runtime import AtlasRuntime
from atlas.trajectory.store import TrajectoryStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--trajectories", nargs="+", default=["data/trajectories/agent_heuristic.jsonl",
                                                          "data/trajectories/agent_llm.jsonl"])
    ap.add_argument("--benchmark", default="data/benchmarks/atlas_papers_v1.jsonl")
    ap.add_argument("--out", default="data/processed/grpo")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    configure_logging("INFO")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    cfg = load_config().with_overrides({"vlm": {"provider": "none"}}) if args.smoke else load_config()
    items = load_benchmark(Path(args.benchmark))
    train_items, _test = split_items(items, 0.3, 0)
    train_ids = {i.id for i in train_items}
    trajs = [t for p in args.trajectories if Path(p).exists() for t in TrajectoryStore(Path(p)).iter()
             if (t.metadata or {}).get("benchmark_id") in train_ids]
    rows = build_grpo_dataset(trajs)
    if not rows:
        print("no usable trajectories (need benchmark_id metadata); run scripts/run_agent_benchmark.py first")
        return 1
    if args.smoke:
        rows = rows[:4]
    rt = AtlasRuntime(cfg)
    env = ResearchEnv(rt.executor, rt.capabilities(), cfg.agent, ExtractiveLLM(), top_k=cfg.retrieval.top_k)
    reward_fn = make_reward_fn(env, {i.id: i for i in items})

    device = resolve_device("auto")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        peft_config = None
    gcfg = GRPOConfig(
        output_dir=args.out, max_steps=2 if args.smoke else args.max_steps, learning_rate=1e-5,
        per_device_train_batch_size=args.num_generations, num_generations=args.num_generations,
        max_completion_length=64 if args.smoke else 128, logging_steps=1, report_to=[], seed=0,
        bf16=device == "cuda", beta=0.04, temperature=1.0, save_strategy="no" if args.smoke else "steps",
    )
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn], args=gcfg, train_dataset=Dataset.from_list(rows),
                          processing_class=tok, peft_config=peft_config)
    trainer.train()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(Path(args.out) / "adapter")
    logs = trainer.state.log_history
    (Path(args.out) / "log_history.json").write_text(json.dumps(logs, indent=2, default=str))
    print(json.dumps(logs[-3:], indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
