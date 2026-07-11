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


def build_vindr_eval_set(vin_csv, positive: str = "lung opacity", negative: str = "no finding",
                         n_max: int | None = None, seed: int = 42):
    """Filter VinDr to a binary opacity/no-finding eval set (normalized match).

    Returns a DataFrame [image_id, label] where label 1 = has opacity, 0 = clean.
    An image with any opacity box is positive; a purely 'no finding' image is
    negative. Raises if either class is empty (usually a label-string mismatch).

    ``n_max`` caps each class to n_max/2 (seeded sample) to bound VinDr caching +
    inference cost; None = all.
    """
    df = vin_csv if isinstance(vin_csv, pd.DataFrame) else pd.read_csv(vin_csv)
    cls = df["class_name"].astype(str).str.strip().str.lower()
    logger.info("VinDr class_name values: %s", sorted(cls.unique()))

    pos_ids = sorted(set(df.loc[cls == positive.lower(), "image_id"]))
    neg_ids = sorted(set(df.loc[cls == negative.lower(), "image_id"]) - set(pos_ids))
    if not pos_ids or not neg_ids:
        raise ValueError(
            f"VinDr filter empty: positive={positive!r} -> {len(pos_ids)} imgs, "
            f"negative={negative!r} -> {len(neg_ids)} imgs. Check class_name spelling "
            f"against the printed unique values."
        )
    if n_max is not None:
        rng = np.random.default_rng(seed)
        cap = max(1, n_max // 2)
        pos_ids = list(rng.permutation(pos_ids)[:cap])
        neg_ids = list(rng.permutation(neg_ids)[:cap])
    rows = [(i, 1) for i in pos_ids] + [(i, 0) for i in neg_ids]
    return pd.DataFrame(rows, columns=["image_id", "label"])


def _cache_vindr_pngs(image_ids, dicom_dir, cache_dir, png_size: int):
    """DICOM->PNG for VinDr eval ids (reuses RSNA read_dicom). Returns {id: png_path}.

    VinDr mirrors ship .dicom or .dcm (some .png/.jpg); we try each. Corrupt or
    missing files are skipped with a warning so external eval never hard-fails
    on one bad image. Kaggle-only (needs cv2/pydicom).
    """
    from pathlib import Path

    import cv2

    from ..data.dicom import read_dicom
    from ..utils.paths import ensure_dir

    dicom_dir, cache_dir = Path(dicom_dir), ensure_dir(cache_dir)
    out = {}
    for iid in image_ids:
        dst = cache_dir / f"{iid}.png"
        if dst.exists():
            out[iid] = str(dst)
            continue
        src = next((dicom_dir / f"{iid}{ext}" for ext in (".dicom", ".dcm", ".png", ".jpg")
                    if (dicom_dir / f"{iid}{ext}").exists()), None)
        if src is None:
            logger.warning("VinDr image %s not found in %s, skipping", iid, dicom_dir)
            continue
        try:
            arr = read_dicom(src) if src.suffix in (".dicom", ".dcm") else cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(dst), cv2.resize(arr, (png_size, png_size)))
            out[iid] = str(dst)
        except Exception as e:  # noqa: BLE001 -- skip one bad image, keep the eval set
            logger.warning("VinDr decode failed for %s: %s", iid, e)
    return out


def _auroc(scores, labels) -> float:
    """Rank-based AUROC (Mann-Whitney), no sklearn dependency."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def external_validate(model, vin_eval, dicom_dir, cfg, cache_dir):
    """Evaluate on VinDr: per-image triage score -> AUROC + calibration drift.

    Triage score = max box confidence per image (screening proxy). Caches the
    eval-set DICOMs to PNG first (VinDr ships DICOM, not PNG). ECE here is
    image-level (triage score vs label) with NO recalibration on VinDr -- that's
    the point: it measures how far RSNA-fit confidence drifts under domain shift.
    Model-bound -> Kaggle only.

    Returns {"auroc", "ece", "n", "scores"}. n = images actually scored.
    """
    from ..calibration.reliability import ece_score
    from ..models.predict import predict_boxes  # lazy

    id_to_png = _cache_vindr_pngs(vin_eval["image_id"], dicom_dir, cache_dir, cfg.png_size)
    scored = vin_eval[vin_eval["image_id"].isin(id_to_png)]
    if len(scored) == 0:
        return {"auroc": float("nan"), "ece": float("nan"), "n": 0, "scores": np.zeros((0,))}

    paths = [id_to_png[i] for i in scored["image_id"]]
    preds = predict_boxes(model, paths, imgsz=cfg.png_size)
    scores = np.array([p["scores"].max() if p["scores"].size else 0.0 for p in preds])
    labels = scored["label"].to_numpy()
    ece = ece_score(scores, labels.astype(float), cfg.n_bins) if scores.size else float("nan")
    return {"auroc": _auroc(scores, labels), "ece": float(ece), "n": int(len(labels)), "scores": scores}
