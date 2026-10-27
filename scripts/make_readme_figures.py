"""Build the README figures from committed benchmark files.

Reads ``benchmarks/results/*/summary.json`` and ``records.jsonl``. Does not
embed hand-typed metrics. Accuracy, Recall@K, and tool calls are means over
per-item records (the same reduction as ``scripts/make_results_table.py``).
p50 latency is ``numpy.percentile`` of record latencies, and must match the
``latency_p50_ms`` already stored on each summary.

Writes:
  docs/assets/accuracy_latency.png
  docs/assets/recall_at_5.png
  docs/assets/social-preview.png
  docs/assets/architecture.svg
  docs/assets/figure_manifest.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
ASSETS = ROOT / "docs" / "assets"

# Display names only. Every plotted value is loaded from RESULTS.
LADDER: list[tuple[str, str, str]] = [
    ("base_llm", "Closed-book", "closed"),
    ("extractive_rag", "Extractive", "extractive"),
    ("vanilla_rag", "Vanilla RAG", "rag"),
    ("hybrid_rag", "Hybrid RAG", "rag"),
    ("hybrid_rerank_rag", "Hybrid + rerank", "rag"),
    ("agent_text_only", "Agent, text-only", "agent"),
    ("agent_heuristic", "Agent, heuristic", "agent"),
    ("agent_llm", "Agent, LLM router", "agent"),
    ("agent_learned", "Agent, RL router", "agent"),
    ("agent_single_search", "Agent, single search", "agent"),
]
RETRIEVAL_ORDER = [
    "retrieval_dense",
    "retrieval_bm25",
    "retrieval_hybrid",
    "retrieval_hybrid_weighted",
    "retrieval_dense_rerank",
    "retrieval_hybrid_rerank",
]
# Same bootstrap as scripts/make_results_table.py so chart intervals match the table.
N_BOOT = 5000
BOOT_SEED = 0

INK = "#1c1917"
MUTED = "#57534e"
FAINT = "#a8a29e"
GRID = "#e7e5e4"
PAPER = "#ffffff"
FAMILY_COLOR = {
    "closed": "#78716c",
    "extractive": "#b45309",
    "rag": "#1d4ed8",
    "agent": "#0f766e",
}
FONT_DIR = Path("/usr/share/fonts/truetype/macos")
FONT_REGULAR = FONT_DIR / "Inter-Regular.ttf"
FONT_MEDIUM = FONT_DIR / "Inter-Medium.ttf"
FONT_SEMIBOLD = FONT_DIR / "Inter-SemiBold.ttf"
FONT_BOLD = FONT_DIR / "Inter-Bold.ttf"


def _register_fonts() -> None:
    for path in (FONT_REGULAR, FONT_MEDIUM, FONT_SEMIBOLD, FONT_BOLD):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams["font.family"] = "Inter"
    plt.rcParams["axes.unicode_minus"] = False


def _load(name: str) -> tuple[dict, list[dict]]:
    folder = RESULTS / name
    summary_path = folder / "summary.json"
    records_path = folder / "records.jsonl"
    if not summary_path.exists() or not records_path.exists():
        raise SystemExit(f"missing benchmark files for {name} under {folder}")
    summary = json.loads(summary_path.read_text())
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    n = summary.get("num_items")
    if len(records) != n or summary["metrics"].get("n") != n:
        raise SystemExit(f"{name}: record count {len(records)} != summary n {n}")
    return summary, records


def _mean(records: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in records if isinstance(r.get(key), int | float)]
    if not vals:
        return None
    return float(np.mean(vals))


def _require_close(name: str, field: str, got: float, expected: float, tol: float = 1e-9) -> None:
    if not math.isclose(got, expected, rel_tol=0, abs_tol=tol):
        raise SystemExit(f"{name}: computed {field} {got} != summary {expected}")


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(N_BOOT, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _p50_ms(records: list[dict], summary: dict, name: str) -> float:
    lat = [float(r["latency_ms"]) for r in records]
    got = float(np.percentile(np.asarray(lat, dtype=float), 50))
    expected = float(summary["metrics"]["latency_p50_ms"])
    _require_close(name, "latency_p50_ms", got, expected, tol=1e-6)
    return got


def _ladder_rows() -> list[dict]:
    rows = []
    for name, label, family in LADDER:
        summary, records = _load(name)
        acc_vals = [float(r["correct"]) for r in records]
        acc = float(np.mean(acc_vals))
        _require_close(name, "correct", acc, float(summary["metrics"]["correct"]))
        lo, hi = _bootstrap_ci(acc_vals)
        tools = _mean(records, "tool_calls")
        if tools is not None and "tool_calls" in summary["metrics"]:
            _require_close(name, "tool_calls", tools, float(summary["metrics"]["tool_calls"]), tol=1e-9)
        rows.append({
            "id": name,
            "label": label,
            "family": family,
            "description": summary["experiment"].get("description", ""),
            "model": (summary.get("models") or {}).get("llm"),
            "n": len(records),
            "accuracy": acc,
            "ci_low": lo,
            "ci_high": hi,
            "latency_p50_s": _p50_ms(records, summary, name) / 1000.0,
            "tool_calls": tools,
            "device": (summary.get("environment") or {}).get("device"),
            "platform": (summary.get("environment") or {}).get("platform"),
            "timestamp": (summary.get("environment") or {}).get("timestamp"),
        })
    return rows


def _retrieval_rows() -> list[dict]:
    rows = []
    for name in RETRIEVAL_ORDER:
        summary, records = _load(name)
        recall = _mean(records, "recall@5")
        if recall is None:
            raise SystemExit(f"{name}: records have no recall@5")
        _require_close(name, "recall@5", recall, float(summary["metrics"]["recall@5"]))
        rows.append({
            "id": name,
            "description": summary["experiment"]["description"],
            "n": len(records),
            "recall_at_5": recall,
            "latency_p50_ms": _p50_ms(records, summary, name),
        })
    return rows


def _omitted_note(ladder_model: str | None) -> str:
    """Footnote for systems that exist on disk but use a different generator."""
    bits = []
    for name in ("agent_base05", "agent_sft"):
        summary, records = _load(name)
        model = (summary.get("models") or {}).get("llm") or "unknown model"
        if model == ladder_model:
            continue
        acc = float(np.mean([float(r["correct"]) for r in records]))
        _require_close(name, "correct", acc, float(summary["metrics"]["correct"]))
        bits.append(f"{name} ({model}) {100 * acc:.1f}%")
    if not bits:
        return ""
    return "Not plotted (different base model): " + "; ".join(bits) + ". Full table in docs/experiments/results.md."


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(PAPER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def write_accuracy_chart(rows: list[dict], note: str, path: Path) -> None:
    _register_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.7), dpi=160, gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(PAPER)
    ax, ax2 = axes
    _style_ax(ax)
    _style_ax(ax2)

    handles = []
    for row in rows:
        color = FAMILY_COLOR[row["family"]]
        marker = _MARKER[row["id"]]
        yerr = np.array([[100 * (row["accuracy"] - row["ci_low"])], [100 * (row["ci_high"] - row["accuracy"])]])
        ax.errorbar(
            row["latency_p50_s"],
            100 * row["accuracy"],
            yerr=yerr,
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=3,
            markersize=8,
            markeredgecolor=PAPER,
            markeredgewidth=0.6,
            zorder=3,
        )
        handles.append(plt.Line2D(
            [0], [0], marker=marker, color="none", markerfacecolor=color, markeredgecolor=color, markersize=7, label=row["label"],
        ))

    ax.set_xlabel("p50 latency (seconds)")
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, 36)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.6, which="major")
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.045))

    agents = [r for r in rows if r["tool_calls"] is not None]
    labels = [r["label"] for r in agents][::-1]
    calls = [r["tool_calls"] for r in agents][::-1]
    colors = [FAMILY_COLOR[r["family"]] for r in agents][::-1]
    y = np.arange(len(agents))
    ax2.barh(y, calls, color=colors, height=0.62, zorder=3)
    ax2.set_yticks(y, labels)
    ax2.set_xlabel("Mean tool calls")
    ax2.set_xlim(0, max(calls) * 1.22)
    ax2.grid(axis="x", color=GRID, linewidth=0.8)
    ax2.grid(axis="y", visible=False)
    for i, value in enumerate(calls):
        ax2.text(value + 0.08, i, f"{value:.2f}", va="center", ha="left", fontsize=8, color=MUTED)

    n = rows[0]["n"]
    device = rows[0]["device"] or "unknown device"
    fig.suptitle("Answer accuracy, latency, and tool calls", fontsize=14, color=INK, x=0.02, ha="left", fontweight="medium")
    fig.text(
        0.02,
        0.91,
        f"{n}-question PDF-verified benchmark · device {device} · bars are 95% bootstrap CIs · one question = {100 / n:.1f} points",
        fontsize=8.5,
        color=MUTED,
    )
    if note:
        fig.text(0.02, 0.012, note, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.10, 1, 0.88))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


_MARKER = {
    "base_llm": "o",
    "extractive_rag": "s",
    "vanilla_rag": "o",
    "hybrid_rag": "^",
    "hybrid_rerank_rag": "D",
    "agent_text_only": "o",
    "agent_heuristic": "^",
    "agent_llm": "s",
    "agent_learned": "D",
    "agent_single_search": "P",
}


def write_recall_chart(rows: list[dict], path: Path) -> None:
    _register_fonts()
    fig, ax = plt.subplots(figsize=(11.4, 4.15), dpi=160)
    fig.patch.set_facecolor(PAPER)
    _style_ax(ax)
    labels = [r["description"] for r in rows][::-1]
    values = [100 * r["recall_at_5"] for r in rows][::-1]
    color = ["#1d4ed8" if "rerank" in r["id"] else "#0f766e" if "hybrid" in r["id"] else "#78716c" for r in rows][::-1]
    y = np.arange(len(rows))
    ax.barh(y, values, color=color, height=0.62, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Recall@5 (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for i, value in enumerate(values):
        ax.text(value + 0.8, i, f"{value:.1f}", va="center", ha="left", fontsize=9, color=INK)
    n = rows[0]["n"]
    ax.set_title(f"Retrieval Recall@5  ·  {n} questions, page-level relevance", loc="left", fontsize=13, color=INK, pad=12)
    fig.text(
        0.01,
        0.01,
        "Means of per-item recall@5 in benchmarks/results/retrieval_*/records.jsonl. Order matches docs/experiments/results.md.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def write_social_preview(ladder: list[dict], retrieval: list[dict], path: Path) -> None:
    width, height = 1280, 640
    image = Image.new("RGB", (width, height), "#f7f6f3")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 14, height), fill="#1d4ed8")

    title_font = _font(FONT_BOLD, 78)
    sub_font = _font(FONT_REGULAR, 28)
    meta_font = _font(FONT_MEDIUM, 18)
    label_font = _font(FONT_MEDIUM, 18)

    draw.text((64, 58), "Atlas", font=title_font, fill=INK)
    tagline = "Research questions over papers, with retrieval, tools, and page-level citations."
    y = 168
    for line in _wrap(draw, tagline, sub_font, 1080):
        draw.text((64, y), line, font=sub_font, fill="#292524")
        y += 38
    second = "Each run is a trajectory for supervised fine-tuning and reinforcement learning."
    for line in _wrap(draw, second, sub_font, 1080):
        draw.text((64, y + 6), line, font=sub_font, fill=MUTED)
        y += 38

    by_id = {row["id"]: row for row in ladder}
    rag_ids = [row["id"] for row in ladder if row["family"] == "rag"]
    agent_ids = [row["id"] for row in ladder if row["family"] == "agent"]
    best_rag = max(rag_ids, key=lambda i: by_id[i]["accuracy"])
    best_agent = max(agent_ids, key=lambda i: by_id[i]["accuracy"])
    rag_acc = by_id[best_rag]["accuracy"]
    agent_acc = by_id[best_agent]["accuracy"]
    closed = by_id["base_llm"]["accuracy"]
    recall_row = next(r for r in retrieval if r["id"] == "retrieval_hybrid_rerank")

    if math.isclose(rag_acc, agent_acc, abs_tol=1e-12):
        headline = f"{by_id[best_rag]['label']} and {by_id[best_agent]['label']} tie"
    elif rag_acc > agent_acc:
        headline = by_id[best_rag]["label"]
        best_shown = rag_acc
    else:
        headline = by_id[best_agent]["label"]
        best_shown = agent_acc
    if math.isclose(rag_acc, agent_acc, abs_tol=1e-12):
        best_shown = rag_acc

    n = by_id["base_llm"]["n"]
    device = by_id["base_llm"]["device"] or "?"
    when = (by_id["base_llm"]["timestamp"] or "")[:10]
    stats = [
        (f"{100 * best_shown:.1f}%", "Answer accuracy", headline),
        (f"{100 * recall_row['recall_at_5']:.1f}", "Recall@5", "Hybrid RRF + cross-encoder"),
        (f"{100 * closed:.1f}% → {100 * rag_acc:.1f}%", "Closed-book → best RAG", by_id[best_rag]["label"]),
    ]
    card_y = 400
    gap = 24
    card_w = (width - 64 - 56 - 2 * gap) // 3
    inner = card_w - 44
    num_size = 54
    while num_size > 28 and any(draw.textlength(number, font=_font(FONT_BOLD, num_size)) > inner for number, _, _ in stats):
        num_size -= 2
    fitted_num = _font(FONT_BOLD, num_size)
    for i, (number, label, detail) in enumerate(stats):
        x = 64 + i * (card_w + gap)
        draw.rounded_rectangle((x, card_y, x + card_w, card_y + 176), radius=16, fill="#ffffff", outline=GRID, width=2)
        draw.text((x + 22, card_y + 28), number, font=fitted_num, fill=INK)
        draw.text((x + 22, card_y + 100), label, font=label_font, fill=MUTED)
        detail_size = 16
        while detail_size > 12 and draw.textlength(detail, font=_font(FONT_REGULAR, detail_size)) > inner:
            detail_size -= 1
        draw.text((x + 22, card_y + 130), detail, font=_font(FONT_REGULAR, detail_size), fill=FAINT)

    footer = f"{n} PDF-verified questions · device {device} · measured {when} · no stars, coverage, or unrun systems"
    draw.text((64, 592), footer, font=meta_font, fill=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")
    saved = Image.open(path)
    if saved.size != (1280, 640):
        raise SystemExit(f"social preview is {saved.size}, expected 1280x640")


def _xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_architecture_svg(path: Path) -> None:
    """Static export of the LangGraph flow in src/atlas/agents/graph.py.

    Edges match ``_build``: START→planner→router, router→act|planner|writer,
    act→critic→router, writer→END. Budget exhaustion is inside ``router_node``
    (it forces the answer action). ``query_sql`` and ``read_document`` are
    executor tools and are not router actions.
    """
    w, h = 1280, 680
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">',
        "<title>Atlas agent and retrieval flow</title>",
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">',
        '<path d="M 0 1.2 L 10 5 L 0 8.8 z" fill="#57534e"/></marker>',
        '<marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">',
        '<path d="M 0 1.2 L 10 5 L 0 8.8 z" fill="#1d4ed8"/></marker>',
        "</defs>",
        f'<rect width="{w}" height="{h}" fill="#f7f6f3"/>',
        '<style>text{font-family:Inter,ui-sans-serif,sans-serif}</style>',
    ]

    def panel(x, y, pw, ph, title):
        parts.append(f'<rect x="{x}" y="{y}" width="{pw}" height="{ph}" rx="16" fill="#ffffff" stroke="#e7e5e4"/>')
        parts.append(f'<text x="{x + 18}" y="{y + 28}" fill="#78716c" font-size="12" font-weight="600" letter-spacing="0.08em">{_xml(title)}</text>')

    def box(x, y, bw, bh, title, sub="", fill="#f7f6f3", stroke="#e7e5e4", title_fill="#1c1917"):
        parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="10" fill="{fill}" stroke="{stroke}"/>')
        parts.append(f'<text x="{x + 12}" y="{y + 22}" fill="{title_fill}" font-size="13" font-weight="600">{_xml(title)}</text>')
        if sub:
            parts.append(f'<text x="{x + 12}" y="{y + 40}" fill="#57534e" font-size="11">{_xml(sub)}</text>')

    def arrow(x1, y1, x2, y2, color="#57534e", marker="arrow"):
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.4" marker-end="url(#{marker})"/>'
        )

    panel(20, 16, 360, 456, "KNOWLEDGE")
    box(40, 56, 320, 52, "PDF, Markdown, TXT", "arXiv papers in the benchmark")
    box(40, 128, 320, 52, "PyMuPDF ingest", "text layer · ruled tables · figure crops")
    box(40, 200, 320, 52, "Page-bounded chunks", "sentence packing, overlap, one page each")
    box(40, 272, 98, 58, "Text", "chunks")
    box(151, 272, 98, 58, "Tables", "SQLite rows")
    box(262, 272, 98, 58, "Figures", "PNG + caption")
    box(40, 354, 320, 92, "Indexed for retrieval", "BM25 postings and bge vectors. Chunk row i is vector i.")

    arrow(200, 108, 200, 128)
    arrow(200, 180, 200, 200)
    arrow(200, 252, 200, 272)
    arrow(89, 330, 160, 354)
    arrow(200, 330, 200, 354)
    arrow(311, 330, 240, 354)

    panel(396, 16, 360, 456, "RETRIEVAL")
    box(416, 56, 150, 64, "BM25", "exact tokens", fill="#f0fdfa", stroke="#99f6e4", title_fill="#0f766e")
    box(586, 56, 150, 64, "Dense", "bge · FAISS / Qdrant", fill="#eff6ff", stroke="#bfdbfe", title_fill="#1d4ed8")
    box(416, 156, 320, 58, "Fusion", "RRF (default) or weighted min-max")
    box(416, 248, 320, 58, "Cross-encoder rerank", "optional · top 30 candidates → top 5")
    box(416, 340, 320, 106, "search_documents", "The only retriever entry point for RAG and the agent. Metadata filters apply before scoring.")

    arrow(491, 120, 540, 156)
    arrow(661, 120, 612, 156)
    arrow(576, 214, 576, 248)
    arrow(576, 306, 576, 340)

    panel(772, 16, 488, 456, "LANGGRAPH AGENT")
    box(860, 48, 220, 48, "planner", "at most 3 sub-questions")
    box(860, 128, 220, 52, "router", "heuristic · LLM JSON · learned")
    box(1112, 128, 128, 52, "writer", "answer or budget")
    box(860, 214, 220, 64, "act", "one tool, then the critic")
    box(860, 312, 220, 52, "critic", "loops back to the router")
    box(800, 388, 430, 64, "Router chooses the edge", "tool → act · replan → planner · answer or budget → writer")

    arrow(970, 96, 970, 128)
    arrow(1080, 154, 1112, 154, color="#0f766e", marker="arrow")
    arrow(970, 180, 970, 214)
    arrow(970, 278, 970, 312)
    parts.append(
        '<path d="M 860 338 L 804 338 L 804 170 L 860 170" fill="none" stroke="#0f766e" stroke-width="1.4" marker-end="url(#arrow)"/>'
    )
    parts.append(
        '<path d="M 860 142 L 784 142 L 784 72 L 860 72" fill="none" stroke="#0f766e" stroke-width="1.4" marker-end="url(#arrow)"/>'
    )
    parts.append('<text x="808" y="260" fill="#0f766e" font-size="11">critic</text>')
    parts.append('<text x="788" y="108" fill="#0f766e" font-size="11">replan</text>')

    parts.append('<rect x="20" y="488" width="1240" height="172" rx="16" fill="#ffffff" stroke="#e7e5e4"/>')
    parts.append('<text x="38" y="516" fill="#78716c" font-size="12" font-weight="600" letter-spacing="0.08em">TRAJECTORY → SFT / RL</text>')
    box(40, 534, 200, 64, "Every agent run", "trajectory JSONL")
    box(270, 534, 200, 64, "Quality filter", "correct + valid citations")
    box(500, 534, 250, 64, "SFT LoRA", "verified, small scale · QLoRA not validated")
    box(780, 534, 250, 64, "REINFORCE router", "verified, small scale · GRPO smoke only")
    arrow(240, 566, 270, 566)
    arrow(470, 566, 500, 566)
    arrow(750, 566, 780, 566)
    parts.append('<text x="40" y="612" fill="#57534e" font-size="11">Adapter and learned policy load into the router. read_document and query_sql are registered tools, not router actions. search_web is off by default.</text>')

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    ladder = _ladder_rows()
    retrieval = _retrieval_rows()
    # Extractive RAG is model-free by design. Every other headline row is the same generator.
    generated = [row for row in ladder if row["family"] != "extractive"]
    models = {row["model"] for row in generated}
    if len(models) != 1:
        raise SystemExit(f"headline ladder mixes generators: {models}")
    note = _omitted_note(next(iter(models)))
    write_accuracy_chart(ladder, note, ASSETS / "accuracy_latency.png")
    write_recall_chart(retrieval, ASSETS / "recall_at_5.png")
    write_social_preview(ladder, retrieval, ASSETS / "social-preview.png")
    write_architecture_svg(ASSETS / "architecture.svg")
    manifest = {
        "source": "benchmarks/results/*/records.jsonl and summary.json",
        "generator": "scripts/make_readme_figures.py",
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "matches": "scripts/make_results_table.py"},
        "ladder": ladder,
        "retrieval": retrieval,
        "omitted_note": note,
    }
    (ASSETS / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {ASSETS / 'accuracy_latency.png'}")
    print(f"wrote {ASSETS / 'recall_at_5.png'}")
    print(f"wrote {ASSETS / 'social-preview.png'}")
    print(f"wrote {ASSETS / 'architecture.svg'}")
    print(note)


if __name__ == "__main__":
    main()
