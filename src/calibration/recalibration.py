"""Label-free recalibration + referral transfer under domain shift.

The finding this attacks: a detector trained on RSNA keeps its *ranking* on VinDr
(AUROC ~0.82) but its *confidence* is badly miscalibrated (ECE 0.09 -> 0.48). You
have NO VinDr labels at deploy time, so you cannot fit a temperature on the target.

This module recalibrates the target triage scores using only the labelled *source*
(RSNA) and the *unlabelled* target scores, then measures whether a referral
operating point chosen on the source transfers safely to the target.

Methods (all monotone in score, so AUROC/risk-coverage rank is unchanged -- the
gain is in calibrated confidence and in threshold transfer, not in ranking):

- ``none``            : raw target scores (the drift baseline).
- ``source_transfer`` : fit temperature T on source, apply the same T to target.
- ``dm``              : Distribution-Matched temperature -- fit source T, then pick a
                        *target* temperature (LABEL-FREE) so the target confidence
                        *distribution* (quantiles) matches the source's calibrated
                        confidence distribution. Recovers the drifted confidence scale
                        without any target label. (Mean-matching alone is degenerate
                        for triage scores symmetric about 0.5 -- over-confidence shifts
                        the spread, not the mean -- so we match the whole distribution.)

ponytail: ``dm`` assumes covariate shift only rescales confidence, not the underlying
accuracy distribution (source and target share a similar calibrated-confidence shape).
That is the honest, simple first method; the residual ECE it leaves IS a reportable
result. Upgrade path if it underperforms: weighted conformal prediction
(Tibshirani 2019) or BBSE label-shift correction -- both heavier.
"""

from __future__ import annotations

import numpy as np

from .reliability import ece_score
from .temperature_scaling import apply_temperature, fit_temperature

METHODS = ("none", "source_transfer", "dm")


def dm_temperature(source_conf, source_correct, target_conf, n_q: int = 50) -> float:
    """Label-free target temperature: match the target confidence DISTRIBUTION to the
    source's calibrated one (quantile L2), via the same coarse-to-fine 1-D scan.

    Source labels only fit the source temperature; the target side sees no labels,
    only its own scores. Quantile matching (not mean matching) so over-confidence,
    which widens the spread while leaving the mean near 0.5, is actually corrected.
    """
    tc = np.asarray(target_conf, float)
    if tc.size == 0:
        return 1.0
    Ts = fit_temperature(source_conf, source_correct)
    q = np.linspace(0.0, 1.0, n_q)
    src_q = np.quantile(apply_temperature(source_conf, Ts), q)

    def obj(T):
        return float(np.mean((np.quantile(apply_temperature(tc, T), q) - src_q) ** 2))

    grid = np.logspace(-1.3, 1.3, 60)  # ~0.05 .. 20, matches fit_temperature
    best = min(grid, key=obj)
    fine = np.linspace(best * 0.6, best * 1.6, 60)
    return float(min(fine, key=obj))


def recalibrate(method: str, source_conf, source_correct, target_conf):
    """Return recalibrated target scores under ``method`` (see module docstring)."""
    tc = np.asarray(target_conf, float)
    if method == "none":
        return tc
    if method == "source_transfer":
        return apply_temperature(tc, fit_temperature(source_conf, source_correct))
    if method == "dm":
        return apply_temperature(tc, dm_temperature(source_conf, source_correct, tc))
    raise ValueError(f"unknown recalibration method {method!r}")


def referral_transfer(source_conf, source_correct, target_conf, target_correct,
                      method: str, target_risk: float = 0.1):
    """Accept target predictions whose RECALIBRATED confidence >= 1 - target_risk.

    The clinical statement: a calibrated triage system sets its abstention threshold
    straight from the risk budget (accept p >= 1-alpha, expect <=alpha error) with no
    target labels. If confidence transferred honestly, realized risk lands near
    ``target_risk``; under drift, raw confidence overshoots (accepts over-confident
    off-domain errors). target_correct is used for EVALUATION only.

    Returns {"threshold", "target_coverage", "realized_risk"}.
    """
    tgt = np.asarray(recalibrate(method, source_conf, source_correct, target_conf), float)
    thr = 1.0 - target_risk
    tcorr = np.asarray(target_correct, float)
    accept = tgt >= thr
    realized = float(1.0 - tcorr[accept].mean()) if accept.any() else float("nan")
    return {"threshold": float(thr), "target_coverage": float(accept.mean()),
            "realized_risk": realized}


def evaluate_recalibration(source_conf, source_correct, target_conf, target_correct,
                           n_bins: int = 15, target_risk: float = 0.1):
    """Compare every method: target ECE + referral-threshold transfer.

    Returns {method: {"ece", "threshold", "target_coverage", "realized_risk"}}.
    Lower target ECE = better calibration; realized_risk closest to target_risk =
    safest threshold transfer.
    """
    tgt_correct = np.asarray(target_correct, float)
    out = {}
    for m in METHODS:
        s = recalibrate(m, source_conf, source_correct, target_conf)
        ece = ece_score(s, tgt_correct, n_bins) if s.size else float("nan")
        ref = referral_transfer(source_conf, source_correct, target_conf, target_correct,
                                m, target_risk)
        out[m] = {"ece": float(ece), **ref}
    return out
