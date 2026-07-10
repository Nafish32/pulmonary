"""coverage_risk_curve, aurc, referral_threshold sanity."""

from __future__ import annotations

import numpy as np

from src.uncertainty.referral import aurc, coverage_risk_curve, referral_threshold


def test_perfect_ranking_low_aurc():
    # confidence perfectly ranks correctness -> accepting top-k stays 0 risk long.
    conf = np.linspace(0.99, 0.01, 100)
    correct = (np.arange(100) < 60).astype(float)  # top-60 correct
    cov, risk = coverage_risk_curve(conf, correct)
    assert cov[0] == 0.01 and risk[0] == 0.0  # most confident is correct
    assert aurc(conf, correct) < aurc(conf, 1 - correct)  # worse ranking = higher AURC


def test_referral_threshold_meets_target():
    conf = np.linspace(0.99, 0.01, 100)
    correct = (np.arange(100) < 60).astype(float)
    thr, cov = referral_threshold(conf, correct, target_risk=0.0)
    # can accept the 60 correct-and-most-confident before hitting an error
    assert np.isclose(cov, 0.60, atol=0.01)
    assert thr > 0.3


def test_referral_impossible_target():
    conf = np.array([0.9, 0.8])
    correct = np.array([0.0, 0.0])  # everything wrong, no set meets 0 risk
    thr, cov = referral_threshold(conf, correct, target_risk=0.0)
    assert thr == float("inf") and cov == 0.0
