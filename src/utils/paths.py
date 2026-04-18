"""Repository-root path resolution; avoid hard-coded absolute paths."""

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (parent of `src/`)."""
    return Path(__file__).resolve().parents[2]


def path_relative_to_repo(*parts: str) -> Path:
    """Join path segments relative to the repository root."""
    return repo_root().joinpath(*parts)
