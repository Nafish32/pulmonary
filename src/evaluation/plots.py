"""Plot helpers. Every plot routes through utils.figures.saved_figure -- no bare savefig."""

from __future__ import annotations

from ..calibration.reliability import ece_score, reliability_bins
from ..utils.figures import saved_figure


def reliability_diagram(confidences, correct, n_bins: int, out_path: str) -> None:
    """Draw + save a reliability diagram (uses the same binning as ece_score)."""
    bins = reliability_bins(confidences, correct, n_bins)
    ece = ece_score(confidences, correct, n_bins)
    with saved_figure(out_path) as fig:
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        if bins:
            conf = [c for c, _, _ in bins]
            acc = [a for _, a, _ in bins]
            ax.plot(conf, acc, "o-", label="model")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(f"Reliability (ECE={ece:.3f}, {n_bins} bins)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
