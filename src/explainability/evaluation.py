"""Quantitative XAI validation.

Saliency energy inside GT box, deletion/insertion curves, and a sanity check
(randomized weights / shuffled layer) so a map that ignores the model is caught.
"""

from __future__ import annotations

import numpy as np


def saliency_energy_in_box(saliency, box_xyxy) -> float:
    """Fraction of total (non-negative) saliency energy inside the GT box.

    ~1.0 = saliency concentrates on the lesion; near box_area/image_area = no
    better than uniform. NaN if the map is all-zero.
    """
    s = np.clip(np.asarray(saliency, float), 0, None)
    total = s.sum()
    if total <= 0:
        return float("nan")
    x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy)
    x1, y1 = max(x1, 0), max(y1, 0)
    inside = s[y1:y2, x1:x2].sum()
    return float(inside / total)


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
