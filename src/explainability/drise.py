"""D-RISE: black-box perturbation saliency for detectors (Petsiuk et al., 2021).

Unlike CAM methods, this needs NO gradients and NO target layer -- it only calls
``predict_boxes``, so it cannot break on an ultralytics version bump (the fragile
part of the CAM path). Mask random regions, weight each mask by how much it
preserves the detection score: regions that matter to the detection get high
weight, so the saliency lands on what the detector actually uses.

Runs through the SAME validation harness as EigenCAM (saliency_energy_in_box,
deletion_curve), so results.md gets a CAM-vs-perturbation comparison for free.
Model-bound -> Kaggle only.
"""

from __future__ import annotations

import numpy as np


def _rise_masks(n, grid, h, w, rng):
    """N smooth RISE masks (n,h,w) in [0,1]: low-res Bernoulli grid, bilinear
    upsample one cell bigger, then a random sub-pixel crop so mask edges don't
    align to a fixed grid (the RISE anti-aliasing trick)."""
    import cv2

    ch, cw = -(-h // grid), -(-w // grid)  # ceil: cell size in full-res pixels
    up_h, up_w = (grid + 1) * ch, (grid + 1) * cw
    cells = (rng.random((n, grid, grid)) < 0.5).astype(np.float32)
    masks = np.empty((n, h, w), np.float32)
    for i in range(n):
        big = cv2.resize(cells[i], (up_w, up_h), interpolation=cv2.INTER_LINEAR)
        y, x = int(rng.integers(0, ch)), int(rng.integers(0, cw))
        masks[i] = big[y:y + h, x:x + w]
    return masks


def d_rise(model, image, imgsz, n_masks: int = 300, grid: int = 8, seed: int = 0):
    """Saliency (H,W) for the detector's top detection on grayscale ``image``.

    Weights each mask by the max detection score of the masked image, then sums
    weighted masks -> pixels whose presence keeps the detection confident score
    high. ``saliency_energy_in_box`` clips + normalizes downstream, so raw
    weighted-sum scale here is fine. ``target_layer`` is intentionally absent:
    D-RISE is black-box.
    """
    from src.models.predict import predict_boxes  # lazy: needs a live model

    h, w = image.shape[:2]
    rng = np.random.default_rng(seed)
    masks = _rise_masks(n_masks, grid, h, w, rng)
    masked = [(image.astype(np.float32) * m).astype(np.uint8) for m in masks]
    preds = predict_boxes(model, masked, imgsz=imgsz)
    scores = np.array([p["scores"].max() if p["scores"].size else 0.0 for p in preds])
    if not scores.any():  # detector saw nothing under any mask -> no signal
        return np.zeros((h, w), np.float32)
    return (scores[:, None, None] * masks).sum(0)
