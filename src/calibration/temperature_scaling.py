"""Temperature scaling for triage confidence.

Confidences are probabilities in (0,1), so we scale in logit space: z = logit(p),
p' = sigmoid(z / T), and pick T>0 minimizing NLL vs the TP/FP labels. Pure-numpy
coarse-to-fine scan -- no torch/scipy dependency for a 1-D convex-ish fit.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _nll(T, z, y):
    p = 1.0 / (1.0 + np.exp(-z / T))
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def apply_temperature(confidences, T: float):
    """Recalibrate probabilities through temperature ``T``."""
    p = np.clip(np.asarray(confidences, float), _EPS, 1.0 - _EPS)
    z = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-z / T))


def fit_temperature(confidences, correct) -> float:
    """Fit a single temperature T on (conf, correct) pairs. Returns T (>0).

    T>1 softens over-confident scores, T<1 sharpens under-confident ones.
    """
    p = np.clip(np.asarray(confidences, float), _EPS, 1.0 - _EPS)
    y = np.asarray(correct, float)
    if p.size == 0:
        return 1.0
    z = np.log(p / (1.0 - p))

    # coarse log-space scan, then refine around the best point.
    grid = np.logspace(-1.3, 1.3, 60)  # ~0.05 .. 20
    best = min(grid, key=lambda t: _nll(t, z, y))
    fine = np.linspace(best * 0.6, best * 1.6, 60)
    best = min(fine, key=lambda t: _nll(t, z, y))
    return float(best)
    # ponytail: 1-D scan, not LBFGS. Fine for one scalar; swap to scipy.minimize_scalar
    # only if the calibration curve ever looks quantized.
