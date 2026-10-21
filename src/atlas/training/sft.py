"""Supervised fine-tuning with LoRA / QLoRA (Hugging Face Transformers + PEFT + TRL).

LoRA: freeze the base weights W and learn a low-rank update W + (alpha/r) * B A
with A in R^{r x d}, B in R^{d x r}. With r=16 on the attention/MLP projections of
a 0.5B model, ~1% of parameters are trained, so fine-tuning fits on a laptop
GPU/MPS and the adapter is a few MB.

QLoRA: additionally load the frozen base in 4-bit NF4 (bitsandbytes) and train
LoRA adapters in bf16 on top. Cuts base-weight memory ~4x (a 7B model trains in
~10 GB). bitsandbytes 4-bit requires CUDA; on other devices `qlora: true` fails
fast with a clear error instead of silently training something else.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from atlas.common.device import resolve_device
from atlas.common.errors import ConfigError
from atlas.observability.logging import get_logger

log = get_logger("sft")


class LoRASettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class SFTSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    trajectories: list[str] = ["data/trajectories/agent_heuristic.jsonl"]
    output_dir: str = "data/processed/sft"
    qlora: bool = False
    lora: LoRASettings = LoRASettings()
    learning_rate: float = 2e-4
    num_train_epochs: float = 2.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_length: int = 2048
    val_fraction: float = 0.2
    min_answer_score: float = 1.0
    include_writer: bool = True
    seed: int = 0
    logging_steps: int = 5
    warmup_steps: int = 3
    device: str = "auto"
    # Exclude trajectories of held-out benchmark questions (same split as RL) to avoid leakage
    # when the SFT agent is later evaluated on the benchmark.
    benchmark: str | None = "data/benchmarks/atlas_papers_v1.jsonl"
    heldout_fraction: float = 0.3
    split_seed: int = 0

    @classmethod
    def from_yaml(cls, path: Path, **overrides: Any) -> SFTSettings:
        data = yaml.safe_load(path.read_text()) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(data)


def load_base_model(settings: SFTSettings):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(settings.device)
    tok = AutoTokenizer.from_pretrained(settings.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if settings.qlora:
        if device != "cuda":
            raise ConfigError("QLoRA (4-bit bitsandbytes) requires a CUDA GPU. Set qlora: false for LoRA on "
                              f"'{device}', or run on a CUDA host (see docs/design/post_training.md).")
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(settings.base_model, quantization_config=bnb, device_map="auto")
        model = prepare_model_for_kbit_training(model)
    else:
        # fp32 on MPS/CPU for numerically stable training; bf16 on CUDA.
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(settings.base_model, dtype=dtype)
    return model, tok, device


def build_datasets(settings: SFTSettings):
    from datasets import Dataset

    from atlas.training.sft_data import split_by_trajectory, trajectories_to_examples
    from atlas.trajectory.store import TrajectoryStore, quality_filter

    keep = quality_filter(min_score=settings.min_answer_score)
    heldout: set[str] = set()
    if settings.benchmark and Path(settings.benchmark).exists():
        from atlas.evaluation.benchmark import load_benchmark, split_items

        _, test = split_items(load_benchmark(Path(settings.benchmark)), settings.heldout_fraction, settings.split_seed)
        heldout = {i.id for i in test}
    trajs = [t for p in settings.trajectories if Path(p).exists() for t in TrajectoryStore(Path(p)).iter()
             if keep(t) and (t.metadata or {}).get("benchmark_id") not in heldout]
    if not trajs:
        raise ConfigError(f"No trajectories passed the quality filter in {settings.trajectories}. "
                          "Collect some with `python scripts/run_agent_benchmark.py`.")
    examples = trajectories_to_examples(trajs, settings.include_writer)
    train, val = split_by_trajectory(examples, settings.val_fraction, settings.seed)
    cols = ["prompt", "completion"]
    stats = {"trajectories": len(trajs), "heldout_questions_excluded": len(heldout), "examples": len(examples), "train": len(train), "val": len(val),
             "router_examples": sum(e["kind"] == "router" for e in examples),
             "writer_examples": sum(e["kind"] == "writer" for e in examples)}
    return (Dataset.from_list([{c: e[c] for c in cols} for e in train]),
            Dataset.from_list([{c: e[c] for c in cols} for e in val]) if val else None, train, val, stats)


def train_sft(settings: SFTSettings) -> dict[str, Any]:
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    train_ds, val_ds, train_raw, val_raw, stats = build_datasets(settings)
    log.info("sft data", extra=stats)
    model, tok, device = load_base_model(settings)
    out = Path(settings.output_dir)
    peft_cfg = LoraConfig(r=settings.lora.r, lora_alpha=settings.lora.alpha, lora_dropout=settings.lora.dropout,
                          target_modules=settings.lora.target_modules, task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir=str(out / "checkpoints"), learning_rate=settings.learning_rate,
        num_train_epochs=settings.num_train_epochs, max_steps=settings.max_steps,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps, max_length=settings.max_length,
        logging_steps=settings.logging_steps, save_strategy="epoch", eval_strategy="epoch" if val_ds else "no",
        seed=settings.seed, report_to=[], bf16=device == "cuda", completion_only_loss=True,
        lr_scheduler_type="cosine", warmup_steps=settings.warmup_steps,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok,
                         peft_config=peft_cfg)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    t0 = time.perf_counter()
    train_out = trainer.train()
    metrics: dict[str, Any] = {"train_loss": train_out.training_loss, "train_seconds": time.perf_counter() - t0,
                               "trainable_params": trainable, "total_params": total,
                               "trainable_pct": 100 * trainable / total, "device": device,
                               "base_model": settings.base_model, "qlora": settings.qlora, **stats}
    if val_ds is not None:
        metrics.update({f"val_{k}": v for k, v in trainer.evaluate().items()})
    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    metrics["log_history"] = trainer.state.log_history
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (out / "val_examples.jsonl").write_text("\n".join(json.dumps(e) for e in val_raw))
    (out / "settings.json").write_text(settings.model_dump_json(indent=2))
    return metrics
