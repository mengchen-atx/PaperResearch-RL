"""Build docs/experiments/results.md from benchmarks/results/*/records.jsonl.

All numbers are recomputed from per-item records (never typed by hand), with 95%
bootstrap confidence intervals for accuracy — with ~46 questions, one question is
~2.2 points, so small differences between systems are often within noise.

Conditional metrics:
  citation validity  mean over answers that contain >=1 citation marker
  citation precision mean over answers that contain >=1 *valid* citation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
OUT = ROOT / "docs" / "experiments" / "results.md"

RETRIEVAL_ORDER = ["retrieval_dense", "retrieval_bm25", "retrieval_hybrid", "retrieval_hybrid_weighted",
                   "retrieval_dense_rerank", "retrieval_hybrid_rerank"]
LADDER_ORDER = ["base_llm", "extractive_rag", "vanilla_rag", "hybrid_rag", "hybrid_rerank_rag",
                "agent_text_only", "agent_heuristic", "agent_llm", "agent_learned", "agent_single_search", "agent_base05", "agent_sft"]


def load(name: str, root: Path = RESULTS) -> tuple[dict, list[dict]] | None:
    d = root / name
    if not (d / "summary.json").exists():
        return None
    recs = [json.loads(line) for line in (d / "records.jsonl").read_text().splitlines() if line.strip()]
    return json.loads((d / "summary.json").read_text()), recs


def mean_where(recs, key, cond=lambda r: True):
    vals = [float(r[key]) for r in recs if cond(r) and isinstance(r.get(key), int | float)]
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def bootstrap_ci(values, n_boot=5000, seed=0):
    if not values:
        return None, None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pct(v):
    return "–" if v is None else f"{100 * v:.1f}"


def ms(v):
    return "–" if v is None else (f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms")


def retrieval_table(root: Path) -> list[str]:
    lines = ["| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@5 | p50 latency |",
             "|---|---|---|---|---|---|---|---|"]
    for name in RETRIEVAL_ORDER:
        got = load(name, root)
        if not got:
            continue
        s, recs = got
        vals = [mean_where(recs, k)[0] for k in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@5")]
        lat = s["metrics"].get("latency_p50_ms")
        lines.append(f"| {s['experiment']['description']} | " + " | ".join(pct(v) for v in vals) + f" | {ms(lat)} |")
    return lines


def ladder_table(root: Path, order=LADDER_ORDER) -> list[str]:
    lines = ["| System | n | Answer acc. (95% CI) | Grounded | Cites (any valid) | Citation validity | "
             "Citation precision | Tool calls | p50 / p95 latency | Tokens / query |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in order:
        got = load(name, root)
        if not got:
            continue
        s, recs = got
        acc = [float(r["correct"]) for r in recs]
        lo, hi = bootstrap_ci(acc)
        grounded = mean_where(recs, "groundedness", lambda r: r.get("groundedness") is not None)[0]
        cites = mean_where(recs, "has_valid_citation")[0]
        validity = mean_where(recs, "citation_validity", lambda r: (r.get("citation_count") or 0) > 0)[0]
        precision = mean_where(recs, "citation_precision", lambda r: (r.get("has_valid_citation") or 0) > 0)[0]
        tools = mean_where(recs, "tool_calls")[0]
        tokens = mean_where(recs, "prompt_tokens")[0], mean_where(recs, "completion_tokens")[0]
        tok = "–" if tokens[0] is None else f"{tokens[0] + (tokens[1] or 0):.0f}"
        m = s["metrics"]
        lines.append(
            f"| {name} | {len(recs)} | {pct(np.mean(acc))} ({pct(lo)}–{pct(hi)}) | {pct(grounded)} | {pct(cites)} | "
            f"{pct(validity)} | {pct(precision)} | {'–' if tools is None else f'{tools:.2f}'} | "
            f"{ms(m.get('latency_p50_ms'))} / {ms(m.get('latency_p95_ms'))} | {tok} |")
    return lines


AGENT_ORDER = ["agent_text_only", "agent_heuristic", "agent_llm", "agent_learned", "agent_single_search", "agent_base05", "agent_sft"]


def agent_table(root: Path) -> list[str]:
    lines = ["| Agent | Success | Tool calls | Unproductive calls | Failed calls | Replans | Policy fallbacks | "
             "Required-tool recall | Budget exhausted |", "|---|---|---|---|---|---|---|---|---|"]
    for name in AGENT_ORDER:
        got = load(name, root)
        if not got:
            continue
        _, recs = got

        def m(k, cond=lambda r: True, recs=recs):
            return mean_where(recs, k, cond)[0]

        rtr = m("required_tool_recall", lambda r: r.get("required_tool_recall") is not None)
        lines.append(f"| {name} | {pct(m('correct'))} | {m('tool_calls'):.2f} | {m('unproductive_tool_calls'):.2f} | "
                     f"{m('failed_tool_calls'):.2f} | {m('replans'):.2f} | {m('policy_fallbacks'):.2f} | {pct(rtr)} | "
                     f"{pct(m('budget_exhausted'))} |")
    return lines


def heldout_ids() -> set[str]:
    sys.path.insert(0, str(ROOT / "src"))
    from atlas.evaluation.benchmark import load_benchmark, split_items

    _, test = split_items(load_benchmark(ROOT / "data/benchmarks/atlas_papers_v1.jsonl"), 0.3, 0)
    return {i.id for i in test}


def heldout_table(root: Path, order=LADDER_ORDER) -> list[str]:
    ids = heldout_ids()
    lines = ["| System | n (held-out) | Answer acc. (95% CI) | Tool calls |", "|---|---|---|---|"]
    for name in order:
        got = load(name, root)
        if not got:
            continue
        recs = [r for r in got[1] if r["id"] in ids]
        acc = [float(r["correct"]) for r in recs]
        lo, hi = bootstrap_ci(acc)
        tools = mean_where(recs, "tool_calls")[0]
        lines.append(f"| {name} | {len(recs)} | {pct(np.mean(acc))} ({pct(lo)}–{pct(hi)}) | "
                     f"{'–' if tools is None else f'{tools:.2f}'} |")
    return lines


def by_type_table(root: Path, order=LADDER_ORDER) -> list[str]:
    types = ["text", "table", "compute", "figure", "multihop"]
    lines = ["| System | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for name in order:
        got = load(name, root)
        if not got:
            continue
        _, recs = got
        cells = []
        for t in types:
            sub = [float(r["correct"]) for r in recs if r.get("type") == t]
            cells.append(f"{pct(np.mean(sub))} (n={len(sub)})" if sub else "–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def rl_section() -> list[str]:
    path = ROOT / "data" / "processed" / "rl" / "results.json"
    if not path.exists():
        return ["_RL results pending: run `python scripts/train_rl.py`._"]
    r = json.loads(path.read_text())
    lines = [f"Held-out questions: {r['test_questions']} (train: {r['train_questions']}), "
             f"{r['config']['iterations']} iterations x {r['config']['episodes_per_update']} episodes, "
             f"training time {r['train_seconds'] / 60:.1f} min.", "",
             "| Policy (same env, held-out) | Success rate | Mean return | Tool calls / episode | Steps / episode |",
             "|---|---|---|---|---|"]
    for k, v in r["test"].items():
        lines.append(f"| {k} | {pct(v['success_rate'])} | {v['mean_return']:.3f} | {v['mean_tool_calls']:.2f} | "
                     f"{v['mean_steps']:.2f} |")
    return lines


def sft_section() -> list[str]:
    tm, ev = ROOT / "data/processed/sft/train_metrics.json", ROOT / "data/processed/sft/eval.json"
    if not tm.exists():
        return ["_SFT results pending: run `python scripts/train_sft.py`._"]
    m = json.loads(tm.read_text())
    lines = [f"Base model `{m['base_model']}`, LoRA (trainable {m['trainable_params']:,} / {m['total_params']:,} params = "
             f"{m['trainable_pct']:.2f}%), device `{m['device']}`, {m['trajectories']} filtered trajectories -> "
             f"{m['train']} train / {m['val']} val examples, train time {m['train_seconds'] / 60:.1f} min, "
             f"final train loss {m['train_loss']:.3f}" + (f", val loss {m['val_eval_loss']:.3f}" if "val_eval_loss" in m else "")
             + "."]
    if ev.exists():
        e = json.loads(ev.read_text())
        lines += ["", "| Router imitation on held-out trajectories | n | JSON valid | Action accuracy |", "|---|---|---|---|"]
        for k in ("base", "sft"):
            lines.append(f"| {k} | {e[k]['n']} | {pct(e[k]['json_valid_rate'])} | {pct(e[k]['action_accuracy'])} |")
    return lines


def main() -> int:
    env = None
    for name in LADDER_ORDER + RETRIEVAL_ORDER:
        got = load(name)
        if got:
            env = got[0].get("environment")
            break
    md = ["# Experimental results", "",
          "_Generated by `scripts/make_results_table.py` from `benchmarks/results/*/records.jsonl`. "
          "Do not edit by hand._", ""]
    if env:
        md += [f"Environment: `{env.get('platform')}`, device `{env.get('device')}`, run at {env.get('timestamp')}.", ""]
    md += ["Benchmark: `data/benchmarks/atlas_papers_v1.jsonl` — 46 questions over 4 arXiv papers (Transformer, BERT, "
           "RAG, Scaling Laws); gold answers and page-level evidence verified against the PDFs by "
           "`scripts/build_benchmark.py`. One question = 2.2 accuracy points; read differences together with the "
           "confidence intervals.", "",
           "## Retrieval (page-level relevance, 46 questions)", "", *retrieval_table(RESULTS), "",
           "## Ablation ladder (end-to-end answers)", "", *ladder_table(RESULTS), "",
           "## Accuracy by question type", "", *by_type_table(RESULTS), "",
           "## Agent behaviour", "",
           "_Required-tool recall: share of table/figure/compute questions where the matching tool "
           "(lookup_table/query_sql, analyze_image, run_python) was used successfully._", "", *agent_table(RESULTS), "",
           "## Held-out split (13 questions never used for SFT data or RL training)", "",
           "_The SFT and learned (RL) systems must be compared on this split; the full-benchmark numbers for "
           "them include their training questions._", "", *heldout_table(RESULTS), "",
           "## Agentic RL (router policy, REINFORCE with baseline)", "", *rl_section(), "",
           "## SFT (LoRA on filtered agent trajectories)", "", *sft_section(), ""]
    mm = RESULTS / "multimodal"
    if mm.exists():
        md += ["## Multimodal subset (table / compute / figure / multihop questions)", "",
               *ladder_table(mm, ["hybrid_rerank_rag", "agent_text_only", "agent_heuristic"]), ""]
    inf = sorted((RESULTS / "inference").glob("*.json")) if (RESULTS / "inference").exists() else []
    if inf:
        md += ["## Inference serving", "", "| Label | Concurrency | TTFT p50 | Latency p50 / p95 | Decode tok/s | "
               "Throughput (req/s, out tok/s) |", "|---|---|---|---|---|---|"]
        for f in inf:
            r = json.loads(f.read_text())
            md.append(f"| {r.get('label') or f.stem} | {r['concurrency']} | {r['ttft_p50_s'] * 1000:.0f} ms | "
                      f"{r['latency_p50_s']:.2f} s / {r['latency_p95_s']:.2f} s | {r['decode_tokens_per_s_mean']:.1f} | "
                      f"{r['throughput_req_per_s']:.2f}, {r['throughput_output_tokens_per_s']:.1f} |")
        md.append("")
    sysb = sorted((RESULTS / "system").glob("*.json")) if (RESULTS / "system").exists() else []
    if sysb:
        md += ["## API load test", "", "| Endpoint | Requests | Concurrency | Errors | Throughput | p50 | p95 | p99 |",
               "|---|---|---|---|---|---|---|---|"]
        for f in sysb:
            r = json.loads(f.read_text())
            md.append(f"| /{r['endpoint']} | {r['requests']} | {r['concurrency']} | {r['errors']} | "
                      f"{r['throughput_rps']:.1f} req/s | {ms(r['latency_p50_ms'])} | {ms(r['latency_p95_ms'])} | "
                      f"{ms(r['latency_p99_ms'])} |")
        md.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
