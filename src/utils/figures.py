"""Context-managed figure saving -- guarantees plt.close() on every exit path.

The ONLY place in src/ allowed to call savefig. Prevents the matplotlib figure
leaks (H3/M7) where bare plt.savefig() left figures open across dozens of plots.

Usage::

    with saved_figure("outputs/reliability.png") as fig:
        ax = fig.add_subplot(111)
        ax.plot(x, y)
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def saved_figure(path: str | Path, dpi: int = 150, **savefig_kwargs) -> Iterator:
    """Yield a fresh Figure, save it on clean exit, always close it.

    Args:
        path: Destination image path (parent dirs created).
        dpi: Save resolution.
        **savefig_kwargs: Forwarded to ``Figure.savefig``.

    Yields:
        A ``matplotlib.figure.Figure`` to draw on.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    try:
        yield fig
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", **savefig_kwargs)
    finally:
        plt.close(fig)
