"""Inference helpers: predict boxes, load GT boxes as xyxy."""

from __future__ import annotations

import numpy as np


def predict_boxes(model, image_paths, conf: float = 0.001, imgsz: int = 640,
                  batch: int = 8):
    """Run inference. Low default conf so calibration sees all predictions.

    imgsz MUST match the png_size the GT is scaled to (see pipeline._scaled_gt),
    or predictions and GT live in different coordinate frames and every IoU is wrong.

    Returns:
        List (one per image) of dicts: {"boxes": (N,4) xyxy float, "scores": (N,)}.
    """
    # A Python-list source hits ultralytics autocast_list -> LoadPilAndNumpy with
    # bs=len(list): the WHOLE test set forwards as one batch (750 imgs @ 640 ->
    # one 18 GiB conv alloc, T4 OOM). stream=True doesn't help; it only streams
    # results. Chunk the list so each predict() call is a bounded small batch.
    paths = list(image_paths)
    out = []
    for i in range(0, len(paths), batch):
        for r in model.predict(
            paths[i:i + batch], conf=conf, imgsz=imgsz, verbose=False
        ):
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
