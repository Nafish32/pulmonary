"""Uncertainty-aware referral: coverage-risk curve, AURC, abstention thresholds."""

from __future__ import annotations

import numpy as np


def coverage_risk_curve(confidences, correct):
    """Risk-coverage curve: accept most-confident predictions first.

    Returns (coverage, risk): at each prefix of confidence-sorted predictions,
    coverage = fraction accepted, risk = error rate among the accepted. Lower risk
    at a given coverage = better selective prediction.
    """
    conf = np.asarray(confidences, float)
    correct = np.asarray(correct, float)
    n = conf.size
    if n == 0:
        return np.zeros((0,)), np.zeros((0,))
    order = np.argsort(-conf)  # most confident first
    err = 1.0 - correct[order]
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return coverage, risk


def aurc(confidences, correct) -> float:
    """Area under the risk-coverage curve (trapezoid). Lower is better."""
    coverage, risk = coverage_risk_curve(confidences, correct)
    if coverage.size < 2:
        return float("nan")
    # trapezoid inline: np.trapz was removed in numpy 2.0, np.trapezoid renamed.
    return float(np.sum(np.diff(coverage) * (risk[1:] + risk[:-1]) / 2))


def referral_threshold(confidences, correct, target_risk: float):
    """Highest-coverage confidence threshold whose accepted-set risk <= target_risk.

    Returns (threshold, coverage_at_threshold). Refer (abstain on) everything below
    the threshold. Returns (inf, 0.0) if no non-empty accepted set meets the target.
    """
    conf = np.asarray(confidences, float)
    coverage, risk = coverage_risk_curve(conf, correct)
    if coverage.size == 0:
        return float("inf"), 0.0
    ok = np.where(risk <= target_risk)[0]
    if ok.size == 0:
        return float("inf"), 0.0
    k = ok.max()  # largest accepted prefix meeting the risk target
    thr = np.sort(conf)[::-1][k]
    return float(thr), float(coverage[k])
