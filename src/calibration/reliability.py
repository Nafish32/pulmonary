"""ECE, Brier score, reliability diagram. ECE and diagram share cfg.n_bins."""

from __future__ import annotations

import numpy as np


def reliability_bins(confidences, correct, n_bins: int):
    """Per-bin (mean_conf, accuracy, weight) over ``n_bins`` equal-width bins.

    Shared by ece_score and the reliability diagram so they can never disagree on
    binning. Empty bins are dropped.
    """
    conf = np.asarray(confidences, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        out.append((conf[m].mean(), correct[m].mean(), m.mean()))
    return out


def ece_score(confidences, correct, n_bins: int) -> float:
    """Expected calibration error over ``n_bins`` equal-width bins."""
    if len(confidences) == 0:
        return float("nan")
    return float(
        sum(w * abs(acc - c) for c, acc, w in reliability_bins(confidences, correct, n_bins))
    )


def brier_score(confidences, correct) -> float:
    """Brier score = mean((conf - correct)^2)."""
    conf = np.asarray(confidences, float)
    correct = np.asarray(correct, float)
    if conf.size == 0:
        return float("nan")
    return float(np.mean((conf - correct) ** 2))
