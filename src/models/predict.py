"""Inference helpers: predict boxes, load GT boxes as xyxy."""

from __future__ import annotations

import numpy as np


def predict_boxes(model, image_paths, conf: float = 0.001):
    """Run inference. Low default conf so calibration sees all predictions.

    Returns:
        List (one per image) of dicts: {"boxes": (N,4) xyxy float, "scores": (N,)}.
    """
    # stream=True: yield per-image results instead of stacking the whole list into
    # one batch tensor -- a flat list source made ultralytics alloc ~12 GiB at once
    # and OOM'd the T4. Generator keeps memory bounded; the loop below consumes it.
    results = model.predict(list(image_paths), conf=conf, verbose=False, stream=True)
    out = []
    for r in results:
        b = r.boxes
        if b is None or len(b) == 0:
            out.append({"boxes": np.zeros((0, 4)), "scores": np.zeros((0,))})
            continue
        out.append({
            "boxes": b.xyxy.cpu().numpy(),
            "scores": b.conf.cpu().numpy(),
        })
    return out


def load_gt_boxes_xyxy(df, image_id, id_col: str = "patientId"):
    """Return ground-truth boxes for one image as pixel xyxy, shape (N, 4).

    RSNA rows store x,y,width,height (pixels); Target==0 / NaN rows are negatives
    and yield an empty (0, 4) array.
    """
    rows = df[df[id_col] == image_id]
    boxes = []
    for r in rows.itertuples():
        if int(getattr(r, "Target", 0)) != 1 or r.width != r.width:  # NaN check
            continue
        x, y, w, h = float(r.x), float(r.y), float(r.width), float(r.height)
        boxes.append([x, y, x + w, y + h])
    return np.array(boxes, dtype=float) if boxes else np.zeros((0, 4))
