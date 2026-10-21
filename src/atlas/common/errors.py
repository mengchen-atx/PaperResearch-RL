"""Typed exceptions. Each carries a user-facing message; details go to logs."""


class AtlasError(Exception):
    """Base class for all Atlas errors."""


class ConfigError(AtlasError):
    pass


class IngestionError(AtlasError):
    pass


class IndexNotFoundError(AtlasError):
    def __init__(self, path: str) -> None:
        super().__init__(f"No index found at '{path}'. Run `atlas ingest <dir>` first.")


class LLMError(AtlasError):
    pass


class ToolError(AtlasError):
    pass


class UnsafePathError(AtlasError):
    pass


class BudgetExceededError(AtlasError):
    pass
