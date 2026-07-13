"""Quantitative XAI validation.

Saliency energy inside GT box, deletion/insertion curves, and a sanity check
(randomized weights / shuffled layer) so a map that ignores the model is caught.
"""

from __future__ import annotations

import numpy as np


def box_union_mask(shape, boxes_xyxy):
    """Boolean HxW mask, True inside the UNION of GT boxes (xyxy pixels).

    Accepts a single box (4,) or many (N,4) -- RSNA images are often bilateral
    (2 opacities), so saliency must be credited for landing in ANY GT box, not
    just the first. atleast_2d makes the single-box case fall through unchanged.
    """
    m = np.zeros(shape, bool)
    for x1, y1, x2, y2 in np.atleast_2d(np.asarray(boxes_xyxy, float)):
        m[max(int(round(y1)), 0):int(round(y2)), max(int(round(x1)), 0):int(round(x2))] = True
    return m


def saliency_energy_in_box(saliency, boxes_xyxy) -> float:
    """Fraction of total (non-negative) saliency energy inside the GT box(es).

    ~1.0 = saliency concentrates on the lesion(s); near union_area/image_area = no
    better than uniform. Unions all boxes (bilateral cases). NaN if map is all-zero.
    """
    s = np.clip(np.asarray(saliency, float), 0, None)
    total = s.sum()
    if total <= 0:
        return float("nan")
    return float(s[box_union_mask(s.shape, boxes_xyxy)].sum() / total)


def deletion_curve(model, image, saliency, steps: int = 20):
    """Deletion curve: detector score vs fraction of most-salient pixels removed.

    Removing truly important pixels should drop the score fast (small AUC = good
    explanation). Returns (fractions, scores). Model-bound -> Kaggle only.
    """
    from src.models.predict import predict_boxes  # lazy: needs a live model

    order = np.argsort(-np.asarray(saliency, float).ravel())
    fractions = np.linspace(0, 1, steps)
    scores = []
    flat = image.astype(np.float32).copy().ravel()
    for frac in fractions:
        k = int(frac * flat.size)
        buf = flat.copy()
        buf[order[:k]] = 0
        img = buf.reshape(image.shape).astype(np.uint8)
        preds = predict_boxes(model, [img])[0]
        scores.append(float(preds["scores"].max()) if preds["scores"].size else 0.0)
    return fractions, np.array(scores)
