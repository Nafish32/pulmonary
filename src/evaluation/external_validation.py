"""External validation on VinDr filtered set (train RSNA -> eval VinDr).

Positive = 'Lung Opacity', Negative = 'No finding' only. Lowercase-normalize both
sides and print class_name.unique() once; fail fast if the configured label yields
zero rows. Do NOT recalibrate on VinDr for the primary external number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


def build_vindr_eval_set(vin_csv, positive: str = "lung opacity", negative: str = "no finding"):
    """Filter VinDr to a binary opacity/no-finding eval set (normalized match).

    Returns a DataFrame [image_id, label] where label 1 = has opacity, 0 = clean.
    An image with any opacity box is positive; a purely 'no finding' image is
    negative. Raises if either class is empty (usually a label-string mismatch).
    """
    df = vin_csv if isinstance(vin_csv, pd.DataFrame) else pd.read_csv(vin_csv)
    cls = df["class_name"].astype(str).str.strip().str.lower()
    logger.info("VinDr class_name values: %s", sorted(cls.unique()))

    pos_ids = set(df.loc[cls == positive.lower(), "image_id"])
    neg_ids = set(df.loc[cls == negative.lower(), "image_id"]) - pos_ids
    if not pos_ids or not neg_ids:
        raise ValueError(
            f"VinDr filter empty: positive={positive!r} -> {len(pos_ids)} imgs, "
            f"negative={negative!r} -> {len(neg_ids)} imgs. Check class_name spelling "
            f"against the printed unique values."
        )
    rows = [(i, 1) for i in pos_ids] + [(i, 0) for i in neg_ids]
    return pd.DataFrame(rows, columns=["image_id", "label"])


def _auroc(scores, labels) -> float:
    """Rank-based AUROC (Mann-Whitney), no sklearn dependency."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def external_validate(model, vin_eval, image_dir, cfg):
    """Evaluate on VinDr: per-image triage score -> AUROC + calibration drift.

    Triage score = max box confidence per image (screening proxy). Model-bound
    (needs live inference) -> Kaggle only. No recalibration on VinDr.
    """
    from ..models.predict import predict_boxes  # lazy

    from pathlib import Path

    paths = [str(Path(image_dir) / f"{i}.png") for i in vin_eval["image_id"]]
    preds = predict_boxes(model, paths)
    scores = np.array([p["scores"].max() if p["scores"].size else 0.0 for p in preds])
    labels = vin_eval["label"].to_numpy()
    return {"auroc": _auroc(scores, labels), "n": int(len(labels)), "scores": scores}
