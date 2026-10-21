# Design: Post-training (SFT, LoRA, QLoRA)

## Data: trajectories → examples (`training/sft_data.py`)
* Source: evaluated trajectories written by `scripts/run_agent_benchmark.py` (each carries `evaluation` and `success`).
* Filter (`trajectory/store.py::quality_filter`): structurally valid, `success` (correct answer **and** ≥1 valid
  citation), bounded length, answer score ≥ threshold. This is rejection sampling: we keep only behavior worth imitating.
* Router examples: for step t, the exact prompt the router saw — evidence prefix rebuilt from per-step
  `new_evidence` counts — and the JSON action actually taken.
* Writer example: grounded prompt over the writer's evidence (ids stored in trajectory metadata) → final cited answer.
* Split by trajectory id (no steps of one episode on both sides). Benchmark questions used for final agent
  evaluation should come from the held-out split (`evaluation.benchmark.split_items`).

## Training (`training/sft.py`)
TRL `SFTTrainer` with conversational prompt/completion data → loss on completion tokens only; PEFT LoRA on
q/k/v/o and MLP projections (r=16, α=32); cosine schedule; fp32 on MPS/CPU, bf16 on CUDA; adapters saved with the
tokenizer; metrics + log history in `train_metrics.json`.

## QLoRA
`qlora: true` → 4-bit NF4 base with double quantization, bf16 compute, `prepare_model_for_kbit_training`.
bitsandbytes 4-bit kernels need CUDA; on other devices training fails fast with an actionable error.
`configs/sft/qlora.yaml` targets Qwen2.5-7B-Instruct on a single 24 GB GPU.

## Evaluation (`training/sft_eval.py`)
Router imitation on held-out trajectories: JSON-valid rate and action accuracy, base model vs. base + adapter.
End-to-end: `configs/experiments/agent_sft.yaml` runs the agent with the adapter (`llm.adapter_path`).

## Serving the adapter
* Local: `llm.provider=hf_local`, `llm.adapter_path=data/processed/sft/adapter`.
* vLLM: `--enable-lora --lora-modules atlas-sft=/adapters/sft`, then `llm.model=atlas-sft`.
