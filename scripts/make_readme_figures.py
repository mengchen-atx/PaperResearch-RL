"""Build the README figures from files already in the repo.

Reads:
  benchmarks/results/*/summary.json and records.jsonl
  data/benchmarks/atlas_papers_v1.jsonl
  docs/experiments/demo_output.md

Writes PNG/SVG under docs/assets/ plus figure_data.json (every plotted number
and the path it came from). Accuracy intervals use the same bootstrap as
scripts/make_results_table.py (seed 0, 5000 resamples). Nothing is typed in.

    python scripts/make_readme_figures.py
    python scripts/make_readme_figures.py --check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
ASSETS = ROOT / "docs" / "assets"
DEMO_MD = ROOT / "docs" / "experiments" / "demo_output.md"
BENCHMARK = ROOT / "data" / "benchmarks" / "atlas_papers_v1.jsonl"
README = ROOT / "README.md"

# Ablation order used by scripts/make_results_table.py. Not sorted by score.
HEADLINE = [
    "base_llm",
    "extractive_rag",
    "vanilla_rag",
    "hybrid_rag",
    "hybrid_rerank_rag",
    "agent_text_only",
    "agent_heuristic",
    "agent_llm",
    "agent_learned",
    "agent_single_search",
]
RETRIEVAL = [
    "retrieval_dense",
    "retrieval_bm25",
    "retrieval_hybrid",
    "retrieval_hybrid_weighted",
    "retrieval_dense_rerank",
    "retrieval_hybrid_rerank",
]

# Abbreviations of experiment.description for the chart legend. The table under
# the figure prints the description string verbatim.
LEGEND = {
    "base_llm": "Closed-book",
    "extractive_rag": "Extractive, no LLM",
    "vanilla_rag": "Vanilla RAG",
    "hybrid_rag": "Hybrid RAG",
    "hybrid_rerank_rag": "Hybrid + rerank",
    "agent_text_only": "Agent, text only",
    "agent_heuristic": "Agent, heuristic",
    "agent_llm": "Agent, LLM router",
    "agent_learned": "Agent, RL router",
    "agent_single_search": "Agent, one search",
}

# The 0.5B SFT pair is a different generator. It stays in the results doc,
# not on the 1.5B latency chart.
POINT_COLOR = {
    "base_llm": "#78716c",
    "extractive_rag": "#a8a29e",
    "vanilla_rag": "#0e7490",
    "hybrid_rag": "#0f766e",
    "hybrid_rerank_rag": "#115e59",
    "agent_text_only": "#ea580c",
    "agent_heuristic": "#c2410c",
    "agent_llm": "#9a3412",
    "agent_learned": "#1d4ed8",
    "agent_single_search": "#1e3a8a",
}

INK = (28, 25, 23)
MUTED = (87, 83, 78)
LINE = (214, 211, 209)
PAPER = (246, 244, 241)
WHITE = (255, 255, 255)
TEAL = (15, 118, 110)
TEAL_SOFT = (240, 253, 250)
RUST = (154, 52, 18)
RUST_SOFT = (255, 247, 237)
BLUE = (30, 58, 138)
BLUE_SOFT = (239, 246, 255)
STONE = (68, 64, 60)


def _register_fonts() -> tuple[str, str]:
    roots = [
        Path("/usr/share/fonts/truetype/macos"),
        Path("/usr/share/fonts/truetype/jetbrains-mono"),
        Path("/Library/Fonts"),
        Path.home() / "Library/Fonts",
    ]
    for folder in roots:
        if not folder.is_dir():
            continue
        for path in folder.glob("*.ttf"):
            if path.name.startswith(("Inter", "JetBrainsMono", "JetBrains")):
                font_manager.fontManager.addfont(str(path))
    names = {face.name for face in font_manager.fontManager.ttflist}
    sans = "Inter" if "Inter" in names else "DejaVu Sans"
    mono = "JetBrains Mono" if "JetBrains Mono" in names else "DejaVu Sans Mono"
    return sans, mono


SANS, MONO = _register_fonts()


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    weight = "Bold" if bold else "Regular"
    candidates = [
        Path(f"/usr/share/fonts/truetype/macos/Inter-{weight}.ttf"),
        Path(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _mono(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/macos/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return _font(size)


def load_summary(name: str) -> dict:
    path = RESULTS / name / "summary.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    data = json.loads(path.read_text())
    data["_path"] = str(path.relative_to(ROOT))
    return data


def load_records(name: str) -> list[dict]:
    path = RESULTS / name / "records.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap_ci(values: list[float], n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Same interval as scripts/make_results_table.py."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pct(value: float) -> str:
    return f"{100 * value:.1f}"


def latency_label(ms: float) -> str:
    """Match scripts/make_results_table.py, but keep sub-millisecond values visible."""
    if ms >= 1000:
        return f"{ms / 1000:.1f} s"
    if ms >= 1:
        return f"{ms:.0f} ms"
    return f"{ms:.2f} ms"


def headline_rows() -> list[dict]:
    rows = []
    for name in HEADLINE:
        summary = load_summary(name)
        records = load_records(name)
        correct = [float(row["correct"]) for row in records]
        low, high = bootstrap_ci(correct)
        tool_vals = [float(row["tool_calls"]) for row in records if isinstance(row.get("tool_calls"), int | float)]
        tools = float(np.mean(tool_vals)) if tool_vals else None
        metrics = summary["metrics"]
        if "latency_p50_ms" not in metrics or "correct" not in metrics:
            raise SystemExit(f"{name} summary is missing accuracy or p50 latency")
        if abs(float(np.mean(correct)) - float(metrics["correct"])) > 1e-9:
            raise SystemExit(f"{name} summary accuracy does not match its records")
        rows.append({
            "name": name,
            "description": summary["experiment"]["description"],
            "source": summary["_path"],
            "records": str((RESULTS / name / "records.jsonl").relative_to(ROOT)),
            "n": len(records),
            "accuracy": float(np.mean(correct)),
            "ci_low": low,
            "ci_high": high,
            "tool_calls": tools,
            "latency_p50_ms": float(metrics["latency_p50_ms"]),
            "llm": summary.get("models", {}).get("llm"),
            "device": summary.get("environment", {}).get("device"),
            "platform": summary.get("environment", {}).get("platform"),
            "timestamp": summary.get("environment", {}).get("timestamp"),
        })
    return rows


def retrieval_rows() -> list[dict]:
    rows = []
    for name in RETRIEVAL:
        summary = load_summary(name)
        metrics = summary["metrics"]
        for key in ("recall@5", "latency_p50_ms"):
            if key not in metrics:
                raise SystemExit(f"{name} summary is missing {key}")
        rows.append({
            "name": name,
            "description": summary["experiment"]["description"],
            "source": summary["_path"],
            "n": summary["num_items"],
            "recall_at_5": float(metrics["recall@5"]),
            "latency_p50_ms": float(metrics["latency_p50_ms"]),
        })
    return rows


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d6d3d1")
    ax.spines["bottom"].set_color("#d6d3d1")
    ax.tick_params(length=0, pad=6)
    ax.grid(axis="y", color="#e7e5e4", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_accuracy(rows: list[dict], sans: str) -> None:
    plt.rcParams.update({
        "font.family": sans,
        "font.size": 12,
        "axes.unicode_minus": False,
        "text.color": "#1c1917",
        "axes.labelcolor": "#1c1917",
        "xtick.color": "#44403c",
        "ytick.color": "#44403c",
    })
    fig, ax = plt.subplots(figsize=(11.4, 6.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for row in rows:
        seconds = row["latency_p50_ms"] / 1000
        accuracy = 100 * row["accuracy"]
        calls = row["tool_calls"]
        size = 90 if calls is None else 70 + calls * 28
        color = POINT_COLOR[row["name"]]
        low = 100 * row["ci_low"]
        high = 100 * row["ci_high"]
        ax.errorbar(
            seconds, accuracy, yerr=[[accuracy - low], [high - accuracy]],
            fmt="none", ecolor=color, elinewidth=1.3, capsize=3, alpha=0.85, zorder=2,
        )
        marker = "o" if calls is None else "s"
        ax.scatter(
            [seconds], [accuracy], s=size, color=color, marker=marker, zorder=3,
            label=LEGEND[row["name"]], linewidths=0,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.09, 55)
    ax.set_ylim(0, 100)
    ax.set_xticks([0.2, 1, 5, 10, 30])
    ax.set_xticklabels(["0.2 s", "1 s", "5 s", "10 s", "30 s"])
    ax.set_xlabel("p50 latency  (log scale)")
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_title("Accuracy vs latency, 46-question benchmark", loc="left", fontsize=16, pad=14, color="#1c1917")
    _style_axes(ax)
    model = next(row["llm"] for row in rows if row["llm"] and row["llm"] != "extractive")
    device = rows[0]["device"]
    legend = ax.legend(
        loc="upper left", frameon=False, fontsize=9, handletextpad=0.4, borderaxespad=0.2, labelspacing=0.35,
    )
    for handle in legend.legend_handles:
        handle.set_sizes([42])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.01, 0.028, "Circles are single-shot. Squares are agents, sized by tool calls. Light grey is extractive (no LLM).", fontsize=8.5, color="#57534e")
    fig.text(0.01, 0.008, f"Whiskers are 95% bootstrap CIs. Generator {model}. Device {device}. Blue: RL router and the one-search baseline.", fontsize=8.5, color="#57534e")
    fig.savefig(ASSETS / "accuracy_latency.png", dpi=150)
    fig.savefig(ASSETS / "accuracy_latency.svg")
    plt.close(fig)


def plot_recall(rows: list[dict], sans: str) -> None:
    plt.rcParams.update({"font.family": sans, "font.size": 12, "text.color": "#1c1917"})
    labels = [row["description"] for row in rows]
    recall = [100 * row["recall_at_5"] for row in rows]
    latency = [row["latency_p50_ms"] for row in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), dpi=150, gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("white")
    colors = ["#0f766e" if "rerank" not in row["name"] else "#115e59" for row in rows]
    axes[0].barh(y, recall, color=colors, height=0.62)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Recall@5 (%)")
    axes[0].set_title("Page-level Recall@5", loc="left", fontsize=14, color="#1c1917")
    for pos, value in zip(y, recall, strict=True):
        axes[0].text(value + 1.2, pos, f"{value:.1f}", va="center", ha="left", fontsize=10, color="#1c1917")
    axes[1].barh(y, latency, color="#c2410c", height=0.62)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xscale("log")
    axes[1].set_xlabel("p50 latency (log scale)")
    axes[1].set_title("What that recall costs", loc="left", fontsize=14, color="#1c1917")
    for pos, value in zip(y, latency, strict=True):
        axes[1].text(value * 1.15, pos, latency_label(value), va="center", ha="left", fontsize=10, color="#1c1917")
    finite = [value for value in latency if value > 0]
    axes[1].set_xlim(min(finite) * 0.45, max(finite) * 6)
    for ax in axes:
        ax.set_facecolor("white")
        _style_axes(ax)
        ax.grid(axis="x", color="#e7e5e4", linewidth=0.8)
        ax.grid(axis="y", visible=False)
    n = rows[0]["n"]
    fig.suptitle(f"Retrieval variants, {n} questions", x=0.01, ha="left", fontsize=16, color="#1c1917")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ASSETS / "recall_at_5.png", dpi=150)
    fig.savefig(ASSETS / "recall_at_5.svg")
    plt.close(fig)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for word in paragraph.split():
            trial = word if not current else f"{current} {word}"
            if draw.textlength(trial, font=font) <= width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        lines.append(current)
    return lines


def _paste_text(
    draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont,
    fill: tuple[int, int, int], width: int, line_gap: int = 4,
) -> int:
    y = xy[1]
    for line in _wrap(draw, text, font, width):
        draw.text((xy[0], y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def _center_text(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str,
    font: ImageFont.ImageFont, fill: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    heights = [font.size + 3 for _ in lines]
    total = sum(heights) - 3
    y = y0 + (y1 - y0 - total) / 2
    for line, height in zip(lines, heights, strict=True):
        w = draw.textlength(line, font=font)
        if w > (x1 - x0) - 16:
            raise SystemExit(f"label does not fit in its box: {line!r} ({w:.0f}px > {x1 - x0 - 16:.0f}px)")
        draw.text((x0 + (x1 - x0 - w) / 2, y), line, font=font, fill=fill)
        y += height


def _box(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: tuple[int, int, int],
    outline: tuple[int, int, int], font: ImageFont.ImageFont, ink: tuple[int, int, int] = INK, radius: int = 12,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)
    _center_text(draw, box, text, font, ink)


def _arrow(
    draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float],
    fill: tuple[int, int, int] = STONE, width: int = 2, dashed: bool = False,
) -> None:
    if dashed:
        length = math.hypot(end[0] - start[0], end[1] - start[1]) or 1
        ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        pos = 0.0
        while pos < length - 10:
            a = (start[0] + ux * pos, start[1] + uy * pos)
            b = (start[0] + ux * min(pos + 7, length - 10), start[1] + uy * min(pos + 7, length - 10))
            draw.line([a, b], fill=fill, width=width)
            pos += 12
    else:
        draw.line([start, end], fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = 9
    left = (end[0] + head * math.cos(angle + 2.6), end[1] + head * math.sin(angle + 2.6))
    right = (end[0] + head * math.cos(angle - 2.6), end[1] + head * math.sin(angle - 2.6))
    draw.polygon([end, left, right], fill=fill)


def draw_architecture() -> None:
    image = Image.new("RGB", (1120, 900), PAPER)
    draw = ImageDraw.Draw(image)
    title = _font(26, bold=True)
    sub = _font(14)
    label = _font(13, bold=True)
    body = _font(15, bold=True)
    small = _font(14, bold=True)
    tiny = _font(13)
    draw.text((28, 22), "How a question becomes a cited answer", font=title, fill=INK)
    draw.text((28, 58), "Same edges as src/atlas/agents/graph.py. Retrieval follows src/atlas/retrieval/retriever.py.", font=sub, fill=MUTED)

    # --- retrieval --------------------------------------------------------------
    draw.rounded_rectangle((20, 92, 1100, 268), radius=16, fill=WHITE, outline=LINE, width=2)
    draw.text((36, 104), "RETRIEVAL", font=label, fill=TEAL)
    retrieval = [
        (36, 148, 200, 220, "PDF / MD / TXT"),
        (232, 140, 430, 212, "chunks, tables,\nfigure crops"),
        (462, 140, 670, 212, "BM25 and dense\nbge-small, FAISS"),
        (702, 140, 870, 212, "RRF or\nweighted"),
        (902, 140, 1084, 212, "cross-encoder\ntop 5"),
    ]
    for box in retrieval:
        _box(draw, box[:4], box[4], TEAL_SOFT, TEAL, body)
    for left, right in ((200, 232), (430, 462), (670, 702), (870, 902)):
        _arrow(draw, (left + 6, 180), (right - 6, 180), fill=TEAL)
    draw.text((36, 230), "Ruled tables also land in SQLite. Figure crops are anchored on captions. Qdrant is the server adapter.", font=tiny, fill=MUTED)

    # --- agent ------------------------------------------------------------------
    draw.rounded_rectangle((20, 284, 1100, 590), radius=16, fill=WHITE, outline=LINE, width=2)
    draw.text((36, 296), "LANGGRAPH    START → planner → router → act → critic → router", font=label, fill=RUST)
    nodes = [
        (36, 360, 168, 424, "question"),
        (200, 360, 360, 424, "planner"),
        (392, 360, 552, 424, "router"),
        (584, 360, 724, 424, "act"),
        (756, 360, 916, 424, "critic"),
    ]
    for box in nodes:
        fill = WHITE if box[4] == "question" else RUST_SOFT
        _box(draw, box[:4], box[4], fill, RUST, body)
    for left, right in ((168, 200), (360, 392), (552, 584), (724, 756)):
        _arrow(draw, (left + 6, 392), (right - 6, 392), fill=RUST)
    # Critic loops back along the top of the row, into the router.
    draw.line([(836, 360), (836, 338), (472, 338)], fill=RUST, width=2)
    _arrow(draw, (472, 338), (472, 358), fill=RUST)
    draw.text((620, 316), "loop", font=tiny, fill=RUST)
    # Answer drops from the router into the writer.
    _arrow(draw, (472, 424), (472, 468), fill=RUST)
    draw.text((482, 434), "answer", font=tiny, fill=RUST)
    _box(draw, (392, 470, 552, 530), "writer", WHITE, RUST, body)
    _arrow(draw, (552, 500), (590, 500), fill=RUST)
    _box(draw, (592, 470, 820, 530), "cited report", RUST_SOFT, RUST, body)
    draw.text((36, 548), "Replan sends the router back to the planner. Policies: heuristic, LLM JSON, learned MLP, one-search.", font=tiny, fill=MUTED)
    draw.text((36, 568), "Act tools: search_documents, lookup_table, analyze_visual → analyze_image, run_python, search_web (off).", font=tiny, fill=MUTED)

    # --- training ---------------------------------------------------------------
    draw.rounded_rectangle((20, 606, 1100, 868), radius=16, fill=WHITE, outline=LINE, width=2)
    draw.text((36, 618), "AFTER THE RUN", font=label, fill=BLUE)
    train = [
        (36, 668, 210, 732, "agent run"),
        (246, 668, 450, 732, "trajectory JSONL"),
        (486, 668, 680, 732, "quality filter"),
    ]
    for box in train:
        _box(draw, box[:4], box[4], BLUE_SOFT, BLUE, small)
    for left, right in ((210, 246), (450, 486)):
        _arrow(draw, (left + 6, 700), (right - 6, 700), fill=BLUE)
    _box(draw, (720, 656, 1064, 720), "SFT LoRA adapter → local LLM", BLUE_SOFT, BLUE, small)
    _arrow(draw, (680, 688), (718, 688), fill=BLUE)
    _box(draw, (720, 748, 1064, 820), "RL env, same nodes\nREINFORCE → learned router", BLUE_SOFT, BLUE, small)
    _arrow(draw, (583, 732), (720, 784), fill=BLUE)
    draw.text((36, 836), "QLoRA and vLLM are configured, not run. GRPO completed a 2-step smoke test, not a full run.", font=tiny, fill=MUTED)
    image.save(ASSETS / "architecture.png", "PNG")


def draw_hero(rows: list[dict]) -> None:
    image = Image.new("RGB", (1200, 420), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1200, 8), fill=TEAL)
    kicker = _font(14, bold=True)
    title = _font(64, bold=True)
    body = _font(22)
    small = _font(16)
    draw.text((48, 36), "MULTIMODAL AGENTIC RESEARCH", font=kicker, fill=TEAL)
    draw.text((48, 68), "Atlas", font=title, fill=INK)
    _paste_text(
        draw, (48, 156),
        "Research questions over papers, with a tool-using agent, page-level citations, and a measured training loop.",
        body, INK, 640, line_gap=6,
    )
    n = rows[0]["n"]
    device = rows[0]["device"]
    draw.text((48, 280), f"{n} verified questions  ·  local models  ·  device {device}", font=small, fill=MUTED)
    draw.text((48, 312), "Python ≥ 3.11   ·   MIT   ·   no leaderboard claim", font=small, fill=MUTED)

    card = (720, 48, 1152, 372)
    draw.rounded_rectangle(card, radius=18, fill=WHITE, outline=LINE, width=2)
    step = _font(16, bold=True)
    steps = [
        (744, 78, 1128, 132, "1    PDFs  →  chunks, tables, figures"),
        (744, 148, 1128, 202, "2    BM25 + dense  →  rerank"),
        (744, 218, 1128, 272, "3    planner · router · tools · critic"),
        (744, 288, 1128, 342, "4    cited answer  →  SFT / RL data"),
    ]
    fills = [TEAL_SOFT, TEAL_SOFT, RUST_SOFT, BLUE_SOFT]
    outlines = [TEAL, TEAL, RUST, BLUE]
    for spec, fill, outline in zip(steps, fills, outlines, strict=True):
        _box(draw, spec[:4], spec[4], fill, outline, step)
    image.save(ASSETS / "hero.png", "PNG")


def draw_social(rows: list[dict]) -> None:
    image = Image.new("RGB", (1280, 640), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1280, 10), fill=TEAL)
    draw.text((72, 64), "MULTIMODAL AGENTIC RESEARCH", font=_font(18, bold=True), fill=TEAL)
    draw.text((72, 104), "Atlas", font=_font(84, bold=True), fill=INK)
    _paste_text(
        draw, (72, 214),
        "A research agent over papers: hybrid retrieval, a LangGraph tool loop, page-level citations, and the runs that train it.",
        _font(28), INK, 700, line_gap=8,
    )
    by_name = {row["name"]: row for row in rows}
    closed = by_name["base_llm"]
    best = by_name["hybrid_rerank_rag"]
    learned = by_name["agent_learned"]
    one = by_name["agent_single_search"]
    strip = (
        f"Closed-book {pct(closed['accuracy'])}%   →   hybrid + rerank {pct(best['accuracy'])}%\n"
        f"RL router {pct(learned['accuracy'])}% at {learned['tool_calls']:.2f} tool calls, "
        f"same accuracy as one fixed search ({pct(one['accuracy'])}%).\n"
        f"n={closed['n']}  ·  device {closed['device']}  ·  {closed['llm']}"
    )
    draw.rounded_rectangle((72, 400, 820, 568), radius=16, fill=WHITE, outline=LINE, width=2)
    _paste_text(draw, (96, 422), strip, _font(20), INK, 700, line_gap=8)

    draw.rounded_rectangle((868, 120, 1208, 568), radius=18, fill=WHITE, outline=LINE, width=2)
    pills = [
        (892, 156, 1184, 230, "retrieve"),
        (892, 254, 1184, 328, "agent loop"),
        (892, 352, 1184, 426, "cite the page"),
        (892, 450, 1184, 524, "train on the trace"),
    ]
    for spec, fill, outline in zip(pills, [TEAL_SOFT, RUST_SOFT, WHITE, BLUE_SOFT], [TEAL, RUST, RUST, BLUE], strict=True):
        _box(draw, spec[:4], spec[4], fill, outline, _font(22, bold=True))
    image.save(ASSETS / "social-preview.png", "PNG")
    if image.size != (1280, 640):
        raise SystemExit(f"social preview is {image.size}, expected 1280x640")


def _demo_section() -> tuple[str, dict]:
    text = DEMO_MD.read_text()
    marker = "## [figure] "
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"{DEMO_MD} has no figure scenario")
    rest = text[start:]
    nxt = rest.find("\n## ", 1)
    section = (rest if nxt < 0 else rest[:nxt]).strip() + "\n"
    gold = None
    for line in BENCHMARK.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] == "fig-01":
            gold = row
            break
    if gold is None:
        raise SystemExit("benchmark is missing fig-01")
    if gold["question"] not in section:
        raise SystemExit("demo figure scenario is not benchmark item fig-01")
    if gold["answer"] not in section:
        raise SystemExit("demo figure answer does not contain the benchmark gold span")
    header = ""
    for line in text.splitlines():
        if line.startswith("Generated by `scripts/demo.py`"):
            header = line
            break
    if not header:
        raise SystemExit("demo_output.md is missing its generation line")
    return section, {"header": header, "gold": gold["answer"], "pages": gold["evidence"][0]["pages"], "id": gold["id"]}


def draw_demo() -> dict:
    section, meta = _demo_section()
    font = _mono(18)
    measure = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    wrapped: list[str] = []
    for raw in section.splitlines():
        if measure.textlength(raw, font=font) <= 1040:
            wrapped.append(raw)
            continue
        buf = ""
        for word in raw.split(" "):
            trial = word if not buf else buf + " " + word
            if measure.textlength(trial, font=font) <= 1040:
                buf = trial
            else:
                wrapped.append(buf)
                buf = word
        if buf:
            wrapped.append(buf)
    line_h = 26
    cited = [line.strip(" *-") for line in section.splitlines() if line.startswith("- **[") and "p." in line]
    cited_note = cited[0] if cited else "no bold citation line"
    note = (
        f"Benchmark {meta['id']} gold answer is {meta['gold']} "
        f"(pages {meta['pages']}, data/benchmarks/atlas_papers_v1.jsonl). "
        f"Bold citation in this run: {cited_note}."
    )
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    header_lines = _wrap(probe, meta["header"], _font(14), 1056)
    note_lines = _wrap(probe, note, _font(15), 1056)
    header_h = 54 + len(header_lines) * 18 + 16
    footer_h = 36 + len(note_lines) * 22
    height = header_h + 16 + line_h * len(wrapped) + footer_h
    image = Image.new("RGB", (1120, height), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1120, 8), fill=TEAL)
    draw.text((32, 22), "Recorded run, not a mock", font=_font(22, bold=True), fill=INK)
    header_bottom = _paste_text(draw, (32, 54), meta["header"], _font(14), MUTED, 1056, line_gap=3)
    y = header_bottom + 14
    draw.rounded_rectangle((24, y, 1096, y + 20 + line_h * len(wrapped)), radius=14, fill=(28, 32, 38), outline=(28, 32, 38))
    y += 12
    for line in wrapped:
        draw.text((40, y), line, font=font, fill=(236, 233, 228))
        y += line_h
    y += 24
    _paste_text(draw, (32, y), note, _font(15), MUTED, 1056, line_gap=4)
    image.save(ASSETS / "demo_run.png", "PNG")
    return {"source": str(DEMO_MD.relative_to(ROOT)), "benchmark_id": meta["id"], "gold": meta["gold"], "lines": len(wrapped)}


def markdown_table(rows: list[dict]) -> str:
    devices = sorted({row["device"] for row in rows})
    dates = sorted({row["timestamp"][:10] for row in rows})
    span = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"
    lines = [
        "Accuracy is the mean of per-question `correct` in each `records.jsonl`, with a 95% bootstrap CI "
        "(seed 0, 5000 resamples; same procedure as `scripts/make_results_table.py`). "
        "p50 latency is `latency_p50_ms` in `summary.json`. "
        f"Device `{' / '.join(devices)}`. Summary dates {span}. "
        "Each file's platform and timestamp is in `docs/assets/figure_data.json`.",
        "",
        "| System | Answer accuracy (95% CI) | Tool calls | p50 latency |",
        "|---|---|---|---|",
    ]
    for row in rows:
        tools = "–" if row["tool_calls"] is None else f"{row['tool_calls']:.2f}"
        lines.append(
            f"| {row['description']} | {pct(row['accuracy'])}% ({pct(row['ci_low'])}–{pct(row['ci_high'])}) "
            f"| {tools} | {latency_label(row['latency_p50_ms'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_manifest(headline: list[dict], retrieval: list[dict], demo: dict) -> None:
    payload = {
        "headline": headline,
        "retrieval": retrieval,
        "demo": demo,
        "bootstrap": {"n_boot": 5000, "seed": 0, "percentiles": [2.5, 97.5]},
        "note": "Generated by scripts/make_readme_figures.py. Do not edit by hand.",
    }
    (ASSETS / "figure_data.json").write_text(json.dumps(payload, indent=2) + "\n")
    (ASSETS / "headline_table.md").write_text(markdown_table(headline))


def refresh_readme(table: str) -> None:
    if not README.is_file():
        return
    text = README.read_text()
    begin, end = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"
    if begin not in text or end not in text:
        return
    block = (
        f"{begin}\n\n"
        "![Accuracy versus p50 latency. Squares are agents.](docs/assets/accuracy_latency.png)\n\n"
        "![Recall@5 and p50 latency for each retrieval variant.](docs/assets/recall_at_5.png)\n\n"
        + table
        + f"\n{end}\n"
    )
    pre, rest = text.split(begin, 1)
    _, post = rest.split(end, 1)
    README.write_text(pre + block + "\n" + post.lstrip("\n"))


def check(headline: list[dict], retrieval: list[dict]) -> int:
    path = ASSETS / "figure_data.json"
    if not path.is_file():
        print("missing docs/assets/figure_data.json", file=sys.stderr)
        return 1
    saved = json.loads(path.read_text())
    ok = saved.get("headline") == headline and saved.get("retrieval") == retrieval
    required = [
        "hero.png", "social-preview.png", "architecture.png", "demo_run.png",
        "accuracy_latency.png", "accuracy_latency.svg", "recall_at_5.png", "recall_at_5.svg",
        "headline_table.md", "figure_data.json",
    ]
    missing = [name for name in required if not (ASSETS / name).is_file()]
    social = Image.open(ASSETS / "social-preview.png")
    if social.size != (1280, 640):
        missing.append(f"social-preview.png size {social.size}")
    if missing or not ok:
        print("figure check failed", missing or "manifest drift", file=sys.stderr)
        return 1
    print("figure check ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare figure_data.json to the result files")
    args = parser.parse_args()
    headline = headline_rows()
    retrieval = retrieval_rows()
    if any(row["n"] != headline[0]["n"] for row in headline):
        raise SystemExit("headline systems were not run on the same number of questions")
    if args.check:
        return check(headline, retrieval)
    ASSETS.mkdir(parents=True, exist_ok=True)
    plot_accuracy(headline, SANS)
    plot_recall(retrieval, SANS)
    draw_architecture()
    draw_hero(headline)
    draw_social(headline)
    demo = draw_demo()
    write_manifest(headline, retrieval, demo)
    refresh_readme(markdown_table(headline))
    print(f"wrote figures under {ASSETS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
