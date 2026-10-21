"""Filesystem safety helpers.

Any path that originates from a user, an API request, or an LLM tool call must be
resolved through `safe_resolve` so it cannot escape an allowed root directory
(e.g. `../../etc/passwd` or a symlink pointing outside the data dir).
"""

from __future__ import annotations

from pathlib import Path

from atlas.common.errors import UnsafePathError


def safe_resolve(candidate: str | Path, allowed_root: str | Path) -> Path:
    root = Path(allowed_root).resolve()
    raw = Path(candidate)
    target = (raw if raw.is_absolute() else root / raw).resolve()
    if target != root and root not in target.parents:
        raise UnsafePathError(f"Path '{candidate}' is outside the allowed directory.")
    return target


def project_root() -> Path:
    """Repository root (directory containing pyproject.toml), falling back to cwd."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
