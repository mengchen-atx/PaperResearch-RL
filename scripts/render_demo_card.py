"""Render a README card from a real ``atlas ask --json`` or ``atlas research --json`` payload.

The image copies question, answer, and retrieved citations out of the JSON
file. It does not fill in an answer of its own. Pass ``--caption`` for the
run conditions (embedder, model, whether this is the benchmark generator).
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path("/usr/share/fonts/truetype/macos")
MONO = Path("/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf")
MONO_MED = Path("/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Medium.ttf")


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text.replace("\n", " "), width=width) or [""]


def render(payload: dict, caption: str, out: Path) -> None:
    question = str(payload["question"])
    answer = str(payload["answer"])
    model = str(payload.get("model") or "")
    retrieved = list(payload.get("retrieved") or [])
    retrieval_ms = payload.get("retrieval_ms")
    generation_ms = payload.get("generation_ms")

    width = 1200
    pad = 36
    line_h = 22
    title_font = _font(FONT_DIR / "Inter-SemiBold.ttf", 15)
    body_font = _font(MONO, 16)
    meta_font = _font(FONT_DIR / "Inter-Regular.ttf", 14)
    head_font = _font(MONO_MED, 15)

    evidence_lines: list[str] = []
    for i, item in enumerate(retrieved[:4], start=1):
        chunk = item.get("chunk") or {}
        label_bits = [chunk.get("filename") or "?"]
        if chunk.get("page") is not None:
            label_bits.append(f"p.{chunk['page']}")
        section = chunk.get("section")
        score = item.get("score")
        via = item.get("retriever") or ""
        meta = ", ".join(label_bits)
        if section:
            meta += f" · {section}"
        if isinstance(score, int | float):
            meta += f" · score {score:.3f}"
        if via:
            meta += f" · {via}"
        snippet = " ".join(str(chunk.get("text") or "").split())
        evidence_lines.append(f"[{i}] {meta}")
        evidence_lines.extend(f"    {line}" for line in _wrap(snippet, 92)[:2])
    if len(retrieved) > 4:
        evidence_lines.append(f"    … {len(retrieved) - 4} more in the JSON capture")

    answer_lines = _wrap(answer, 96)
    question_lines = _wrap(question, 96)
    blocks = [
        ("Question", question_lines),
        (f"Answer ({model})" if model else "Answer", answer_lines),
        ("Retrieved evidence", evidence_lines or ["(none)"]),
    ]
    caption_lines = _wrap(caption, 110)
    timing = ""
    if isinstance(retrieval_ms, int | float) and isinstance(generation_ms, int | float):
        timing = f"retrieval {retrieval_ms:.0f} ms · generation {generation_ms:.0f} ms"

    height = pad
    height += 28 + len(caption_lines) * 20 + 16
    for _title, lines in blocks:
        height += 26 + len(lines) * line_h + 16
    if timing:
        height += 28
    height += pad

    image = Image.new("RGB", (width, height), "#f7f6f3")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((12, 12, width - 12, height - 12), radius=18, fill="#ffffff", outline="#e7e5e4", width=2)
    y = pad
    draw.text((pad, y), "Atlas · real run", font=title_font, fill="#1c1917")
    y += 28
    for line in caption_lines:
        draw.text((pad, y), line, font=meta_font, fill="#57534e")
        y += 20
    y += 12
    for title, lines in blocks:
        draw.text((pad, y), title, font=head_font, fill="#0f766e")
        y += 26
        for line in lines:
            draw.text((pad, y), line, font=body_font, fill="#1c1917")
            y += line_h
        y += 16
    if timing:
        draw.text((pad, y), timing, font=meta_font, fill="#78716c")
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, "PNG")
    print(f"wrote {out} ({width}x{height})")


def render_research(payload: dict, caption: str, out: Path) -> None:
    question = str(payload["query"])
    answer = str(payload["answer"])
    decisions = list(payload.get("decisions") or [])
    tools = list(payload.get("tool_history") or [])
    sources = list((payload.get("report") or {}).get("sources") or [])
    policy = payload.get("policy") or ""
    status = payload.get("status") or ""

    title_font = _font(FONT_DIR / "Inter-SemiBold.ttf", 15)
    body_font = _font(MONO, 15)
    meta_font = _font(FONT_DIR / "Inter-Regular.ttf", 14)
    head_font = _font(MONO_MED, 15)

    trace = []
    tool_i = 0
    for decision in decisions:
        action = str(decision.get("action") or "")
        line = f"router → {action}"
        if action not in {"answer", "replan"} and tool_i < len(tools):
            rec = tools[tool_i]
            tool_i += 1
            new = rec.get("new_evidence")
            ms = rec.get("latency_ms")
            ok = "ok" if rec.get("ok") else "failed"
            extra = ok
            if isinstance(new, int):
                extra += f", +{new} evidence"
            if isinstance(ms, int | float):
                extra += f", {ms:.0f} ms"
            line += f"  ({extra})"
        trace.append(line)

    source_lines = []
    for source in sources:
        flag = "cited" if source.get("cited") else "not cited"
        label = source.get("label") or "?"
        modality = source.get("modality") or ""
        tool = source.get("tool") or ""
        source_lines.append(f"[{source.get('marker')}] {label}  ({modality} via {tool}, {flag})")
        excerpt = " ".join(str(source.get("excerpt") or "").split())
        source_lines.extend(f"    {line}" for line in _wrap(excerpt, 96)[:2])

    blocks = [
        ("Question", _wrap(question, 96)),
        (f"Trace  ·  policy {policy}  ·  status {status}", trace or ["(none)"]),
        (f"Answer ({payload.get('model') or 'unedited'})", _wrap(answer, 96)),
        ("Sources", source_lines or ["(none)"]),
    ]
    width = 1200
    pad = 36
    line_h = 22
    caption_lines = _wrap(caption, 108)
    height = pad + 28 + len(caption_lines) * 20 + 12
    for _title, lines in blocks:
        height += 26 + len(lines) * line_h + 14
    height += pad

    image = Image.new("RGB", (width, height), "#f7f6f3")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((12, 12, width - 12, height - 12), radius=18, fill="#ffffff", outline="#e7e5e4", width=2)
    y = pad
    draw.text((pad, y), "Atlas · real CPU quickstart", font=title_font, fill="#1c1917")
    y += 28
    for line in caption_lines:
        draw.text((pad, y), line, font=meta_font, fill="#57534e")
        y += 20
    y += 10
    for title, lines in blocks:
        draw.text((pad, y), title, font=head_font, fill="#0f766e")
        y += 26
        for line in lines:
            draw.text((pad, y), line, font=body_font, fill="#1c1917")
            y += line_h
        y += 14
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, "PNG")
    print(f"wrote {out} ({width}x{height})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="JSON from `atlas ask --json` or `atlas research --json`")
    parser.add_argument("out", type=Path)
    parser.add_argument("--caption", required=True, help="How this run was produced. Not an answer.")
    args = parser.parse_args()
    payload = json.loads(args.src.read_text())
    if "query" in payload and "report" in payload:
        render_research(payload, args.caption, args.out)
        return
    if "question" not in payload or "answer" not in payload:
        raise SystemExit("JSON is not an atlas ask or research payload")
    render(payload, args.caption, args.out)


if __name__ == "__main__":
    main()
