"""Row metadata side channel kept separate from tensor inputs."""

from contextvars import ContextVar

evaluation_rows: ContextVar[list[dict[str, str]] | None] = ContextVar(
    "publication_evaluation_rows", default=None
)
