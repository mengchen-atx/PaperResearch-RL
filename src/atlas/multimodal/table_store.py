"""SQLite store for extracted tables -> enables the agent's `query_sql` tool.

Why SQL for tables? Retrieval returns a table as markdown text, which an LLM can
read but cannot reliably compute over (sums, max, ratios). Storing the rows in a
real database lets the agent answer quantitative questions with exact arithmetic.

Schema:
    atlas_tables(table_name, table_id, doc_id, filename, page, caption, columns)
    t_<n>(<sanitized columns...>)      -- one SQL table per extracted table
Numeric-looking cells are stored as REAL so comparisons/aggregations work.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from atlas.common.types import Document
from atlas.multimodal.pdf_extract import ExtractedTable

_NUMERIC = re.compile(r"^[-+]?\d+(\.\d+)?$")


def sanitize_column(name: str, index: int, seen: set[str]) -> str:
    base = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower() or f"col{index + 1}"
    if base[0].isdigit():
        base = f"c_{base}"
    col, n = base[:40], 2
    while col in seen:
        col, n = f"{base[:36]}_{n}", n + 1
    seen.add(col)
    return col


def parse_cell(value: str) -> str | float:
    cleaned = value.replace(",", "").replace("*", "").strip()
    return float(cleaned) if _NUMERIC.match(cleaned) else value


class TableStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def reset(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS atlas_tables (table_name TEXT PRIMARY KEY, table_id TEXT, "
            "doc_id TEXT, filename TEXT, page INTEGER, caption TEXT, columns TEXT)"
        )
        return conn

    def add_tables(self, doc: Document, tables: list[ExtractedTable]) -> int:
        with self._connect() as conn:
            (count,) = conn.execute("SELECT COUNT(*) FROM atlas_tables").fetchone()
            for offset, t in enumerate(tables):
                name = f"t_{count + offset + 1}"
                seen: set[str] = set()
                cols = [sanitize_column(h, i, seen) for i, h in enumerate(t.header)]
                conn.execute(f'CREATE TABLE "{name}" ({", ".join(f"{c!r}" for c in cols)})')
                width = len(cols)
                for row in t.rows:
                    padded = (row + [""] * width)[:width]
                    conn.execute(
                        f'INSERT INTO "{name}" VALUES ({", ".join("?" for _ in cols)})',
                        [parse_cell(v) for v in padded],
                    )
                conn.execute(
                    "INSERT INTO atlas_tables VALUES (?,?,?,?,?,?,?)",
                    (name, t.table_id, doc.doc_id, doc.filename, t.page, t.caption, json.dumps(cols)),
                )
        return len(tables)
