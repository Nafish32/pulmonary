"""Robustness benchmark: mAP degradation + calibration drift across corruptions."""

from __future__ import annotations

from .corruption import CORRUPTIONS, corrupt


def _score(model, images, gts, cfg, imgsz):
    """(mAP@50, ECE) for one image set. ECE on the conf>=0.05 gate, like _evaluate."""
    from src.calibration.reliability import ece_score
    from src.evaluation.metrics import label_tp_fp, map50
    from src.models.predict import predict_boxes

    preds = predict_boxes(model, images, imgsz=imgsz)
    mAP = map50(preds, gts)
    conf, correct = label_tp_fp(preds, gts)
    gate = conf >= 0.05
    ece = ece_score(conf[gate], correct[gate], cfg.n_bins) if gate.any() else float("nan")
    return mAP, ece


def run_robustness(model, images, gts, cfg, severities=(1, 2, 3)):
    """Sweep corruptions x severities; report mAP@50 + ECE vs clean.

    ``images`` = list of uint8 arrays, ``gts`` = matching per-image (N,4) GT boxes.
    Returns {"clean": (mAP, ECE), (kind, severity): (mAP, ECE), ...}. mAP drop =
    accuracy degradation; ECE rise = calibration drift. Model-bound -> Kaggle only.
    """
    imgsz = cfg.png_size  # arrays are png_size square; keep pred frame == GT frame
    out = {"clean": _score(model, images, gts, cfg, imgsz)}
    for kind in CORRUPTIONS:
        for s in severities:
            corrupted = [corrupt(im, kind, s) for im in images]
            out[(kind, s)] = _score(model, corrupted, gts, cfg, imgsz)
    return out
