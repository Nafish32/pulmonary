"""Inference helpers: predict boxes, load GT boxes as xyxy."""

from __future__ import annotations

import numpy as np


def predict_boxes(model, image_paths, conf: float = 0.001, imgsz: int = 640):
    """Run inference. Low default conf so calibration sees all predictions.

    imgsz MUST match the png_size the GT is scaled to (see pipeline._scaled_gt),
    or predictions and GT live in different coordinate frames and every IoU is wrong.

    Returns:
        List (one per image) of dicts: {"boxes": (N,4) xyxy float, "scores": (N,)}.
    """
    # imgsz caps the forward pass: without it ultralytics runs at the PNG's native
    # ~3000px and one conv2d alloc'd ~12 GiB, OOM'ing the T4. stream=True yields
    # per-image so nothing stacks. Both together keep memory bounded.
    results = model.predict(
        list(image_paths), conf=conf, imgsz=imgsz, verbose=False, stream=True
    )
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
