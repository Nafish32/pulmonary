"""Path helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing; return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
