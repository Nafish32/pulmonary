"""Label-free recalibration: DM temperature must cut target ECE + tame threshold transfer."""

from __future__ import annotations

import numpy as np

from src.calibration.recalibration import evaluate_recalibration, dm_temperature
from src.calibration.reliability import ece_score


def _shifted_domains(n=6000, seed=0, shift_T=0.5):
    """Source: perfectly calibrated. Target: same accuracy, confidence over-sharpened
    by a temperature-type shift (the regime dm is designed to correct).

    Target confidence = true prob pushed toward 0/1 via temperature ``shift_T`` < 1
    (overconfident). A label-free temperature ~ 1/shift_T should undo it.
    """
    from src.calibration.temperature_scaling import apply_temperature

    rng = np.random.default_rng(seed)
    src_conf = rng.uniform(0.05, 0.95, n)
    src_correct = (rng.uniform(size=n) < src_conf).astype(float)  # calibrated

    tgt_p = rng.uniform(0.05, 0.95, n)                    # true per-item accuracy
    tgt_correct = (rng.uniform(size=n) < tgt_p).astype(float)
    tgt_conf = apply_temperature(tgt_p, shift_T)          # over-sharpened confidence
    return src_conf, src_correct, tgt_conf, tgt_correct


def test_dm_cuts_target_ece():
    src_conf, src_correct, tgt_conf, tgt_correct = _shifted_domains()
    res = evaluate_recalibration(src_conf, src_correct, tgt_conf, tgt_correct, n_bins=15)

    ece_none = res["none"]["ece"]
    ece_dm = res["dm"]["ece"]
    # dm is label-free yet must meaningfully beat the raw-drift baseline.
    assert ece_dm < 0.7 * ece_none, f"dm ECE {ece_dm:.3f} vs none {ece_none:.3f}"
    # source_transfer barely helps here (source T ~ 1), so dm should also beat it.
    assert ece_dm <= res["source_transfer"]["ece"] + 1e-9


def test_dm_temperature_softens_overconfidence():
    src_conf, src_correct, tgt_conf, _ = _shifted_domains()
    T = dm_temperature(src_conf, src_correct, tgt_conf)
    assert T > 1.0, f"overconfident target needs softening T>1, got {T:.3f}"


def test_dm_shrinks_operating_point_calibration_gap():
    src_conf, src_correct, tgt_conf, tgt_correct = _shifted_domains()
    res = evaluate_recalibration(src_conf, src_correct, tgt_conf, tgt_correct,
                                 n_bins=15, coverage=0.2)
    # Same accepted set across methods (recalibration is rank-preserving), so actual
    # risk is identical -- the confidence CLAIM is what differs.
    assert res["none"]["actual_risk"] == res["dm"]["actual_risk"]
    # Raw over-confidence claims a far-too-low risk at the operating point; dm's claim
    # matches reality, so its calibration gap is much smaller.
    assert res["dm"]["calib_gap"] < 0.5 * res["none"]["calib_gap"]


def test_recalibration_is_rank_preserving():
    # temperature scaling is monotone -> it must not reorder scores (AUROC invariant).
    src_conf, src_correct, tgt_conf, _ = _shifted_domains(n=500)
    from src.calibration.recalibration import recalibrate

    s = recalibrate("dm", src_conf, src_correct, tgt_conf)
    assert np.array_equal(np.argsort(tgt_conf), np.argsort(s))
