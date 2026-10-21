"""run_python: numerical analysis in a restricted subprocess.

SECURITY BOUNDARY — read before enabling:
This is a *development-grade* sandbox with defense in depth, not a hardened
isolation boundary:
  1. Static AST check: only whitelisted imports (math, statistics, ...), no
     dunder attribute access, no exec/eval/open/compile/__import__/getattr...
  2. Separate interpreter process (`python -I`: isolated mode, no user site,
     no env-based injection), empty environment, temporary working directory.
  3. POSIX resource limits: CPU seconds, address space (Linux), file size, no
     child processes (Linux), plus a wall-clock timeout.
In production (`environment: production`) the config validator forces
`tools.python_mode: disabled` unless explicitly overridden. A real multi-tenant
deployment should run this tool in a gVisor/Firecracker/nsjail container.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from typing import ClassVar

from pydantic import Field

from atlas.common.config import ToolsConfig
from atlas.common.errors import ToolError
from atlas.common.ids import sha256_text
from atlas.common.types import Evidence, Modality
from atlas.tools.base import Tool, ToolArgs, ToolResult

ALLOWED_MODULES = frozenset({"math", "statistics", "fractions", "decimal", "itertools", "functools",
                             "collections", "json", "re", "operator"})
FORBIDDEN_NAMES = frozenset({"exec", "eval", "compile", "open", "__import__", "input", "globals", "locals",
                             "vars", "getattr", "setattr", "delattr", "breakpoint", "help", "memoryview",
                             "exit", "quit"})
MAX_CODE_CHARS = 4000
MAX_OUTPUT_CHARS = 4000


def validate_code(code: str) -> None:
    if len(code) > MAX_CODE_CHARS:
        raise ToolError(f"Code too long ({len(code)} > {MAX_CODE_CHARS} chars)")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ToolError(f"Syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name.split(".")[0] not in ALLOWED_MODULES]
            if bad:
                raise ToolError(f"Import not allowed: {bad}. Allowed: {sorted(ALLOWED_MODULES)}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                raise ToolError(f"Import not allowed: {node.module}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ToolError(f"Access to private attribute '{node.attr}' is not allowed")
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise ToolError(f"Use of '{node.id}' is not allowed")


_PRELUDE = """
import resource, sys
def _limit(res, value):
    try:
        resource.setrlimit(res, (value, value))
    except (ValueError, OSError):
        pass  # some limits are unsupported on macOS; the wall-clock timeout still applies
_limit(resource.RLIMIT_CPU, {cpu})
_limit(resource.RLIMIT_FSIZE, 1 << 20)
if sys.platform.startswith("linux"):
    _limit(resource.RLIMIT_AS, {mem} * 1024 * 1024)
    _limit(resource.RLIMIT_NPROC, 0)
del _limit, resource
"""


class RunPythonArgs(ToolArgs):
    code: str = Field(min_length=1, description="Python code; print() the result")
    purpose: str = Field("", max_length=300)


class RunPython(Tool):
    name = "run_python"
    description = ("Execute short Python (math/statistics only, no I/O) for exact numerical analysis. "
                   "print() the final result.")
    Args: ClassVar = RunPythonArgs
    cost = 1.0

    def __init__(self, cfg: ToolsConfig) -> None:
        self.cfg = cfg

    def run(self, args: RunPythonArgs) -> ToolResult:
        if self.cfg.python_mode == "disabled":
            raise ToolError("Python execution is disabled (tools.python_mode=disabled)")
        validate_code(args.code)
        prelude = _PRELUDE.format(cpu=max(1, int(self.cfg.python_timeout_s)), mem=self.cfg.python_memory_mb)
        with tempfile.TemporaryDirectory(prefix="atlas-py-") as tmp:
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-c", prelude + "\n" + args.code],
                    capture_output=True, text=True, timeout=self.cfg.python_timeout_s, cwd=tmp, env={},
                )
            except subprocess.TimeoutExpired as exc:
                raise ToolError(f"Python timed out after {self.cfg.python_timeout_s}s") from exc
        if proc.returncode != 0:
            err = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit code {proc.returncode}"
            return ToolResult(tool=self.name, ok=False, error=err[:500], output=proc.stdout[:MAX_OUTPUT_CHARS])
        out = proc.stdout.strip()[:MAX_OUTPUT_CHARS]
        ev = Evidence(evidence_id=f"py:{sha256_text(args.code)[:12]}", modality=Modality.COMPUTATION,
                      content=f"Computation ({args.purpose or 'analysis'}):\n{args.code}\nResult: {out}",
                      source_tool=self.name, metadata={"stdout": out})
        return ToolResult(tool=self.name, ok=True, output=out, evidence=[ev] if out else [])
