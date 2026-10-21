"""analyze_image: route a question to the VLM with the most relevant figure.

Steps: (1) locate the figure — explicit figure_id/filename, an uploaded image, or
retrieval restricted to FIGURE chunks (captions are indexed); (2) send the
rendered figure + question to the VLM; (3) return the VLM reading as evidence
citing the figure's document/page/figure id.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import Field

from atlas.common.errors import ToolError, UnsafePathError
from atlas.common.paths import safe_resolve
from atlas.common.types import Evidence, Modality
from atlas.multimodal.vlm import VisionLanguageModel
from atlas.retrieval.filters import MetadataFilter
from atlas.retrieval.retriever import Retriever
from atlas.tools.base import Tool, ToolArgs, ToolResult

VLM_INSTRUCTION = ("You are analyzing a figure from a research paper. Caption: {caption}\n"
                   "Question: {question}\nDescribe precisely what the figure shows that is relevant "
                   "(axes, trends, values). Do not guess values you cannot read.")


class AnalyzeImageArgs(ToolArgs):
    question: str = Field(min_length=1, max_length=1000)
    figure_id: str | None = Field(None, description="e.g. 'Figure 1'")
    filename: str | None = None
    image_path: str | None = Field(None, description="Path of an uploaded image (inside the upload dir)")


class AnalyzeImage(Tool):
    name = "analyze_image"
    description = "Answer a question about a figure/chart/image with a vision-language model."
    Args: ClassVar = AnalyzeImageArgs
    cost = 3.0  # VLM calls are the most expensive action

    def __init__(self, vlm: VisionLanguageModel, retriever: Retriever, allowed_roots: list[Path]) -> None:
        self.vlm = vlm
        self.retriever = retriever
        self.allowed_roots = allowed_roots

    def _resolve_image(self, raw: str) -> Path:
        for root in self.allowed_roots:
            try:
                p = safe_resolve(raw, root)
            except UnsafePathError:
                continue
            if p.exists():
                return p
        raise ToolError("Image path is not inside an allowed directory or does not exist")

    def _find_figure(self, args: AnalyzeImageArgs) -> tuple[Path, Evidence]:
        figs = [c for c in self.retriever.index.chunks if c.modality == Modality.FIGURE]
        chosen = None
        if args.figure_id:
            cands = [c for c in figs if c.metadata.get("figure_id", "").lower() == args.figure_id.lower()
                     and (args.filename is None or c.filename == args.filename)]
            chosen = cands[0] if cands else None
        if chosen is None:
            flt = MetadataFilter(modalities=[Modality.FIGURE], filenames=[args.filename] if args.filename else None)
            hits = self.retriever.retrieve(args.question, top_k=1, flt=flt)
            chosen = hits[0].chunk if hits else None
        if chosen is None:
            raise ToolError("No figure found for this question")
        ev = Evidence(evidence_id=chosen.chunk_id, modality=Modality.FIGURE, content=chosen.text,
                      source_tool=self.name, doc_id=chosen.doc_id, filename=chosen.filename, page=chosen.page,
                      metadata={"figure_id": chosen.metadata.get("figure_id", "")})
        return self._resolve_image(chosen.metadata["image_path"]), ev

    def run(self, args: AnalyzeImageArgs) -> ToolResult:
        if args.image_path:
            path = self._resolve_image(args.image_path)
            base = Evidence(evidence_id=f"img:{path.name}", modality=Modality.FIGURE, content="",
                            source_tool=self.name, filename=path.name)
        else:
            path, base = self._find_figure(args)
        caption = base.content or "(uploaded image)"
        resp = self.vlm.analyze(path, VLM_INSTRUCTION.format(caption=caption, question=args.question))
        ev = base.model_copy(update={
            "evidence_id": f"{base.evidence_id}:vlm",
            "content": f"{caption}\nVLM reading ({resp.model}): {resp.text}",
            "metadata": {**base.metadata, "vlm_model": resp.model, "image_path": str(path)}})
        return ToolResult(tool=self.name, ok=True, output=resp.text, evidence=[ev],
                          metadata={"vlm_latency_ms": resp.latency_ms})
