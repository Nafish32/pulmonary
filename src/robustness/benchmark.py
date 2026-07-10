"""Robustness benchmark: mAP degradation + calibration drift across corruptions."""

from __future__ import annotations

from .corruption import CORRUPTIONS, corrupt


def run_robustness(model, images, gts, cfg, severities=(1, 2, 3)):
    """Sweep corruptions x severities; report mAP@50 degradation vs clean.

    ``images`` = list of uint8 arrays, ``gts`` = matching per-image (N,4) GT boxes.
    Returns {"clean": mAP, (kind, severity): mAP, ...}. Model-bound -> Kaggle only.
    """
    from src.evaluation.metrics import map50  # lazy: pulls numpy-heavy eval path
    from src.models.predict import predict_boxes

    out = {"clean": map50(predict_boxes(model, images), gts)}
    for kind in CORRUPTIONS:
        for s in severities:
            corrupted = [corrupt(im, kind, s) for im in images]
            out[(kind, s)] = map50(predict_boxes(model, corrupted), gts)
    return out
