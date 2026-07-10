"""Dataset root discovery. Auto-finds RSNA/VinDr/CheXpert mounts, fails fast.

Kaggle dataset dir names vary run-to-run, so we search by *file markers* (known
CSV/dir names) under ``input_root`` instead of hard-coding mount paths. RSNA is
required (train domain); VinDr + CheXpert are optional (external eval / pretrain).
"""

from __future__ import annotations

from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

# marker -> glob. First match wins. Grounded in the public competition layouts:
#   RSNA Pneumonia Detection: stage_2_train_labels.csv + stage_2_train_images/
#   VinBigData/VinDr-CXR:      train.csv (image_id,class_name,...) + train/ dicoms
_DICOM_EXTS = {".dcm", ".dicom"}


def _first(root: Path, name: str) -> Path | None:
    """First path matching bare filename ``name`` under ``root``.

    Shallow depths first (``root/name``, ``root/*/name`` ...), rglob only as a
    last resort. Kaggle markers sit 1-2 levels below the mount, so this avoids
    descending into the ~26k-file image dirs -- rglob over the whole mount, run
    once per marker, was the multi-minute stall at stage [1/8].
    """
    for depth in range(4):
        matches = sorted(root.glob("*/" * depth + name))
        if matches:
            return matches[0]
    matches = sorted(root.rglob(name))  # fallback: unusually deep layout
    return matches[0] if matches else None


def _dir_has_dicom(d: Path | None) -> bool:
    if d is None or not d.is_dir():
        return False
    for p in d.iterdir():
        if p.suffix.lower() in _DICOM_EXTS:
            return True
    return False


def discover_datasets(input_root: str | Path) -> dict:
    """Resolve dataset roots under a Kaggle input mount.

    Returns:
        Dict of resolved Paths: rsna_csv, rsna_images_dir (required), plus
        vin_csv, vin_images_dir, chexpert_csv (optional -> None if absent).

    Raises:
        FileNotFoundError: input_root missing, or RSNA (required) not found.
    """
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"input_root does not exist: {root}")

    out: dict = {}

    # --- RSNA (required train domain) ---
    rsna_csv = _first(root, "stage_2_train_labels.csv")
    rsna_images = _first(root, "stage_2_train_images")
    if rsna_csv is None or not _dir_has_dicom(rsna_images):
        raise FileNotFoundError(
            f"RSNA not found under {root}. Expected 'stage_2_train_labels.csv' and "
            f"a 'stage_2_train_images/' dir containing .dcm files "
            f"(csv={rsna_csv}, images={rsna_images}). Add the RSNA dataset in the "
            "Kaggle UI (Add data)."
        )
    out["rsna_csv"] = rsna_csv
    out["rsna_images_dir"] = rsna_images

    # --- VinDr / VinBigData (optional external eval) ---
    # shallow scan for any train.csv, keep the one whose mount slug looks like VinDr.
    vin_csv = None
    for cand in sorted(root.glob("*/train.csv")) + sorted(root.glob("*/*/train.csv")):
        if "vin" in str(cand.parent).lower():
            vin_csv = cand
            break
    vin_images = None
    if vin_csv is not None:
        cand = vin_csv.parent / "train"
        vin_images = cand if _dir_has_dicom(cand) else None
    out["vin_csv"] = vin_csv
    out["vin_images_dir"] = vin_images
    if vin_csv is None:
        logger.warning("VinDr not found under %s (external validation will skip)", root)

    # --- CheXpert (optional pretrain, no boxes) ---
    chexpert = None
    for cand in sorted(root.glob("*/train.csv")) + sorted(root.glob("*/*/train.csv")):
        if "chexpert" in str(cand.parent).lower():
            chexpert = cand
            break
    out["chexpert_csv"] = chexpert

    logger.info(
        "discovered: rsna=%s vin=%s chexpert=%s",
        bool(out["rsna_csv"]), bool(out["vin_csv"]), bool(out["chexpert_csv"]),
    )
    return out
