"""Detection + triage metrics.

``safe_mean`` guards the empty-array footgun (M1): compute after a length check,
never inside a short-circuit f-string conditional.
"""

from __future__ import annotations

import numpy as np


def safe_mean(arr) -> float:
    """Mean of ``arr``, or nan if empty (no RuntimeWarning)."""
    arr = np.asarray(arr, dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def iou_xyxy(a, b):
    """Pairwise IoU between boxes ``a`` (N,4) and ``b`` (M,4) -> (N, M)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(union > 0, inter / union, 0.0)


def _match_one(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    """Greedy score-ordered TP/FP labeling for one image.

    Returns (scores_desc, correct_flags, n_matched_gt); each GT matches >=1 pred once.
    """
    ps = np.asarray(pred_scores, float)
    order = np.argsort(-ps)
    ps = ps[order]
    pb = np.asarray(pred_boxes, float).reshape(-1, 4)[order] if len(ps) else np.zeros((0, 4))
    gt = np.asarray(gt_boxes, float).reshape(-1, 4)
    ious = iou_xyxy(pb, gt)
    used = np.zeros(len(gt), bool)
    correct = np.zeros(len(ps))
    for i in range(len(ps)):
        if len(gt) == 0:
            break
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr and not used[j]:
            correct[i] = 1.0
            used[j] = True
    return ps, correct, int(used.sum())


def label_tp_fp(preds, gts, iou_thr=0.5):
    """Flatten all predictions to (confidences, correct) arrays across images.

    Single source for calibration + referral: ``correct[i]`` = 1 if pred i is a TP
    at ``iou_thr``. ``preds`` = per-image {"boxes","scores"}; ``gts`` = per-image (N,4).
    """
    scores, correct = [], []
    for p, g in zip(preds, gts):
        ps, c, _ = _match_one(p["boxes"], p["scores"], g, iou_thr)
        scores.append(ps)
        correct.append(c)
    if not scores:
        return np.zeros((0,)), np.zeros((0,))
    return np.concatenate(scores), np.concatenate(correct)


def map50(preds, gts, iou_thr=0.5) -> float:
    """mAP@50 (single opacity class), VOC all-point AP over the pooled PR curve."""
    total_gt = sum(len(np.asarray(g).reshape(-1, 4)) for g in gts)
    if total_gt == 0:
        return float("nan")
    scores, correct = label_tp_fp(preds, gts, iou_thr)
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores)
    correct = correct[order]
    tp = np.cumsum(correct)
    fp = np.cumsum(1.0 - correct)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):  # monotone envelope
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
