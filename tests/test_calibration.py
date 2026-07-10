"""ECE, Brier, temperature scaling sanity."""

from __future__ import annotations

import numpy as np

from src.calibration.reliability import brier_score, ece_score
from src.calibration.temperature_scaling import apply_temperature, fit_temperature


def test_ece_perfectly_calibrated_is_zero():
    # conf 0.0 all wrong, conf 1.0 all right -> acc matches conf in each bin.
    conf = np.array([0.0] * 50 + [1.0] * 50)
    correct = np.array([0.0] * 50 + [1.0] * 50)
    assert ece_score(conf, correct, n_bins=15) < 1e-9


def test_ece_detects_overconfidence():
    conf = np.full(100, 0.9)
    correct = np.zeros(100)  # always wrong but confident
    assert np.isclose(ece_score(conf, correct, n_bins=15), 0.9, atol=1e-9)


def test_brier_bounds():
    assert brier_score(np.array([1.0]), np.array([1.0])) == 0.0
    assert brier_score(np.array([1.0]), np.array([0.0])) == 1.0


def test_temperature_softens_overconfident():
    rng = np.random.default_rng(0)
    # true prob 0.6, but reported confidence pushed toward 0.95 (over-confident)
    y = (rng.random(2000) < 0.6).astype(float)
    conf = np.where(y == 1, 0.95, 0.9)
    T = fit_temperature(conf, y)
    assert T > 1.0  # needs softening
    # recalibrated mean confidence moves toward the 0.6 accuracy
    assert apply_temperature(conf, T).mean() < conf.mean()


def test_temperature_wellcalibrated_near_one():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.random(4000) < p).astype(float)  # confidence == true prob
    assert 0.7 < fit_temperature(p, y) < 1.4
