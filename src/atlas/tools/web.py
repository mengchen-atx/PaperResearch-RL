"""Web search tool.

Backends:
  * wikipedia — free MediaWiki search API, no key. Good for background facts.
  * tavily    — search API designed for agents; needs TAVILY_API_KEY.
  * disabled  — default for offline/reproducible benchmarks (web results change
                over time, which would make benchmark numbers non-reproducible).
"""

from __future__ import annotations

import re
from typing import ClassVar

import httpx
from pydantic import Field

from atlas.common.config import ToolsConfig, read_secret
from atlas.common.errors import ToolError
from atlas.common.ids import sha256_text
from atlas.common.types import Evidence, Modality
from atlas.tools.base import Tool, ToolArgs, ToolResult

USER_AGENT = "AtlasResearchAgent/0.1 (research prototype; https://github.com/)"


class SearchWebArgs(ToolArgs):
    query: str = Field(min_length=1, max_length=400)


class SearchWeb(Tool):
    name = "search_web"
    description = "Search the public web (Wikipedia or Tavily) for background information not in the local corpus."
    Args: ClassVar = SearchWebArgs
    cost = 2.0

    def __init__(self, cfg: ToolsConfig, transport: httpx.BaseTransport | None = None) -> None:
        self.cfg = cfg
        self._client = httpx.Client(timeout=cfg.web_timeout_s, headers={"User-Agent": USER_AGENT},
                                    transport=transport)

    def run(self, args: SearchWebArgs) -> ToolResult:
        if self.cfg.web_backend == "disabled":
            raise ToolError("Web search is disabled (tools.web_backend=disabled)")
        results = self._wikipedia(args.query) if self.cfg.web_backend == "wikipedia" else self._tavily(args.query)
        evidence = [
            Evidence(evidence_id=f"web:{sha256_text(r['url'])[:12]}", modality=Modality.WEB,
                     content=f"{r['title']}: {r['content']}", source_tool=self.name, url=r["url"])
            for r in results[: self.cfg.web_max_results]
        ]
        return ToolResult(tool=self.name, ok=True, evidence=evidence,
                          output="\n".join(f"- {e.url}: {e.content[:120]}" for e in evidence) or "no results")

    def _wikipedia(self, query: str) -> list[dict]:
        r = self._client.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": query, "format": "json",
            "srlimit": self.cfg.web_max_results})
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
        titles = "|".join(h["title"] for h in hits)
        if not titles:
            return []
        ex = self._client.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "titles": titles,
            "format": "json"})
        ex.raise_for_status()
        pages = ex.json().get("query", {}).get("pages", {}).values()
        extracts = {p["title"]: p.get("extract", "") for p in pages}
        return [{"title": h["title"], "url": "https://en.wikipedia.org/wiki/" + h["title"].replace(" ", "_"),
                 "content": re.sub(r"\s+", " ", extracts.get(h["title"], ""))[:1200]} for h in hits]

    def _tavily(self, query: str) -> list[dict]:
        key = read_secret("TAVILY_API_KEY")
        if not key:
            raise ToolError("TAVILY_API_KEY is not set")
        r = self._client.post("https://api.tavily.com/search", json={
            "api_key": key, "query": query, "max_results": self.cfg.web_max_results})
        r.raise_for_status()
        return [{"title": x.get("title", ""), "url": x["url"], "content": x.get("content", "")}
                for x in r.json().get("results", [])]
