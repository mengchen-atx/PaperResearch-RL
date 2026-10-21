"""SFT (LoRA / QLoRA) on filtered agent trajectories, then evaluate base vs adapter.

    python scripts/train_sft.py -c configs/sft/lora.yaml [--max-steps N] [--no-eval]
    python scripts/train_sft.py -c configs/sft/lora.yaml --prepare-only   # dataset stats only

Outputs in <output_dir>: adapter/ (PEFT weights), train_metrics.json, val_examples.jsonl, eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atlas.observability.logging import configure_logging
from atlas.training.sft import SFTSettings, build_datasets, train_sft
from atlas.training.sft_eval import evaluate_router_imitation, load_val_examples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/sft/lora.yaml")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--eval-max", type=int, default=None)
    args = ap.parse_args()
    configure_logging("INFO")
    settings = SFTSettings.from_yaml(Path(args.config), max_steps=args.max_steps, num_train_epochs=args.epochs)
    if args.prepare_only:
        *_, stats = build_datasets(settings)
        print(json.dumps(stats, indent=2))
        return 0
    metrics = train_sft(settings)
    print(json.dumps({k: v for k, v in metrics.items() if k != "log_history"}, indent=2, default=str))
    if args.no_eval:
        return 0
    out = Path(settings.output_dir)
    val = load_val_examples(out / "val_examples.jsonl")
    base = evaluate_router_imitation(settings.base_model, None, val, args.eval_max)
    tuned = evaluate_router_imitation(settings.base_model, str(out / "adapter"), val, args.eval_max)
    result = {"base": base, "sft": tuned}
    (out / "eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: {m: v[m] for m in ("n", "json_valid_rate", "action_accuracy")} for k, v in result.items()},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
