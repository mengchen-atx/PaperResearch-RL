"""Table tools: query_sql (agent-written SQL) and lookup_table (find a table by topic).

Safety: the database is opened read-only (`mode=ro` URI), only a single SELECT/WITH
statement is accepted, results are row-capped, and a progress handler aborts
runaway queries. Tables come from our own ingestion, but the SQL comes from an
LLM, so it is treated as untrusted input.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import ClassVar

from pydantic import Field

from atlas.common.errors import ToolError
from atlas.common.ids import sha256_text
from atlas.common.types import Evidence, Modality
from atlas.retrieval.bm25 import BM25Index
from atlas.tools.base import Tool, ToolArgs, ToolResult

MAX_ROWS = 50
QUERY_TIMEOUT_S = 3.0
_FORBIDDEN_SQL = re.compile(r"\b(attach|detach|pragma|insert|update|delete|drop|alter|create|replace|vacuum)\b", re.I)


def validate_sql(sql: str) -> str:
    stmt = sql.strip().rstrip(";").strip()
    if ";" in stmt:
        raise ToolError("Only a single SQL statement is allowed")
    if not re.match(r"^(select|with)\b", stmt, re.I):
        raise ToolError("Only SELECT queries are allowed")
    if _FORBIDDEN_SQL.search(stmt):
        raise ToolError("Query contains a forbidden keyword")
    return stmt


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise ToolError("No table database found; ingest PDFs with tables first")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    deadline = time.monotonic() + QUERY_TIMEOUT_S
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    return conn


def table_catalog(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    with _connect_ro(db_path) as conn:
        rows = conn.execute("SELECT table_name, table_id, filename, page, caption, columns FROM atlas_tables").fetchall()
    return [{"table_name": r[0], "table_id": r[1], "filename": r[2], "page": r[3], "caption": r[4],
             "columns": json.loads(r[5])} for r in rows]


def _rows_to_text(cols: list[str], rows: list[tuple]) -> str:
    lines = [" | ".join(cols)] + [" | ".join("" if v is None else str(v) for v in r) for r in rows]
    return "\n".join(lines)


class QuerySQLArgs(ToolArgs):
    sql: str = Field(min_length=6, max_length=2000)


class QuerySQL(Tool):
    name = "query_sql"
    description = ("Run a read-only SQL SELECT over tables extracted from the PDFs. The catalog table "
                   "`atlas_tables(table_name, table_id, filename, page, caption, columns)` lists them.")
    Args: ClassVar = QuerySQLArgs
    cost = 1.0

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def run(self, args: QuerySQLArgs) -> ToolResult:
        stmt = validate_sql(args.sql)
        try:
            with _connect_ro(self.db_path) as conn:
                cur = conn.execute(stmt)
                cols = [d[0] for d in cur.description or []]
                rows = cur.fetchmany(MAX_ROWS)
        except sqlite3.Error as exc:
            return ToolResult(tool=self.name, ok=False, error=f"SQL error: {exc}")
        text = _rows_to_text(cols, rows)
        ev = Evidence(evidence_id=f"sql:{sha256_text(stmt)[:12]}", modality=Modality.TABLE,
                      content=f"SQL: {stmt}\n{text}", source_tool=self.name, metadata={"rows": len(rows)})
        return ToolResult(tool=self.name, ok=True, output=text[:1500], evidence=[ev] if rows else [])


class LookupTableArgs(ToolArgs):
    query: str = Field(min_length=1, max_length=500)
    max_tables: int = Field(1, ge=1, le=3)


class LookupTable(Tool):
    """Find the extracted table(s) most relevant to a question and return their rows.

    This is the table-access action available to policies that cannot write SQL
    (heuristic / RL policies); LLM policies can use `query_sql` directly.
    """

    name = "lookup_table"
    description = "Find the most relevant extracted table for a topic and return its rows with caption and page."
    Args: ClassVar = LookupTableArgs
    cost = 1.0

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def run(self, args: LookupTableArgs) -> ToolResult:
        catalog = table_catalog(self.db_path)
        if not catalog:
            return ToolResult(tool=self.name, ok=False, error="No extracted tables available")
        docs = [f"{t['table_id']} {t['caption']} {' '.join(t['columns'])}" for t in catalog]
        scores = BM25Index().build(docs).scores(args.query)
        order = [int(i) for i in scores.argsort()[::-1] if scores[i] > 0][: args.max_tables]
        if not order:
            return ToolResult(tool=self.name, ok=True, output="no matching table", evidence=[])
        evidence = []
        with _connect_ro(self.db_path) as conn:
            for i in order:
                t = catalog[i]
                cur = conn.execute(f'SELECT * FROM "{t["table_name"]}" LIMIT {MAX_ROWS}')
                text = _rows_to_text([d[0] for d in cur.description], cur.fetchall())
                evidence.append(Evidence(
                    evidence_id=f"table:{t['table_name']}:{t['filename']}", modality=Modality.TABLE,
                    content=f"{t['caption'] or t['table_id']}\n{text}", source_tool=self.name,
                    filename=t["filename"], page=t["page"], score=float(scores[i]),
                    metadata={"table_id": t["table_id"], "sql_table": t["table_name"]}))
        return ToolResult(tool=self.name, ok=True, evidence=evidence,
                          output="\n".join(f"- {e.citation_label()}" for e in evidence))
