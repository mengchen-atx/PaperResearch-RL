"""Robust parsing of JSON emitted by LLMs.

Small open models frequently wrap JSON in code fences, add prose around it, or
leave trailing commas. We extract the first balanced JSON object and apply a few
safe repairs. Callers must still validate the result against a schema and have
a fallback when parsing fails — invalid model output is expected, not exceptional.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _balanced_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = re.sub(r"```(?:json)?", "", text)
    candidate = _balanced_object(text)
    if candidate is None:
        return None
    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
        try:
            value = json.loads(attempt)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            continue
    return None
