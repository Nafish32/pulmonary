"""DICOM -> PNG cache with a disk-quota guard.

Before writing: project cache size (n_images x size_per_image x n_variants) and
assert it fits under 80% of free space on working_root. Default single-channel
raw at 640px -- the raw+CLAHE+unsharp triplet at 1024px is ~75GB vs a 20GB quota.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from src.data.dicom import read_dicom
from src.utils.logger import get_logger
from src.utils.paths import ensure_dir

logger = get_logger(__name__)


def assert_cache_fits(
    n_images: int,
    bytes_per_image: int,
    n_variants: int,
    working_root: str | Path,
    headroom: float = 0.8,
) -> None:
    """Raise if the projected PNG cache would exceed ``headroom`` of free disk."""
    projected = n_images * bytes_per_image * n_variants
    free = shutil.disk_usage(str(working_root)).free
    if projected > headroom * free:
        raise RuntimeError(
            f"projected cache {projected / 1e9:.1f}GB exceeds "
            f"{headroom:.0%} of {free / 1e9:.1f}GB free on {working_root}. "
            "Drop png_size to 640 and cache single-channel raw only."
        )


def build_png_cache(df, images_dir, out_dir, cfg, id_col: str = "patientId"):
    """Parallel DICOM->PNG cache. Returns df with png_path/orig_h/orig_w columns.

    Resizes each frame to ``cfg.png_size`` square (normalized YOLO boxes survive
    the resize; orig_h/orig_w are kept so pixel labels can be normalized). One id
    may have many label rows (RSNA), so we decode each *unique* id once and join
    the metadata back onto every row.
    """
    images_dir, out_dir = Path(images_dir), ensure_dir(out_dir)
    ids = df[id_col].unique().tolist()
    # single-channel uint8 at png_size square; single variant (raw only).
    # Only project PNGs not already cached -- a warm cache (2nd detector reusing the
    # same png dir) has already spent that disk, so counting it double-trips the guard.
    missing = [p for p in ids if not (cfg.cache_png and (out_dir / f"{p}.png").exists())]
    assert_cache_fits(len(missing), cfg.png_size * cfg.png_size, 1, cfg.working_root)

    def _one(pid):
        dst = out_dir / f"{pid}.png"
        try:
            arr = read_dicom(images_dir / f"{pid}.dcm")  # decode gives orig dims
        except RuntimeError as e:
            logger.warning("skipping corrupt/unreadable DICOM %s: %s", pid, e)
            return pid, None, None, None
        h, w = arr.shape
        if not (cfg.cache_png and dst.exists()):
            cv2.imwrite(str(dst), cv2.resize(arr, (cfg.png_size, cfg.png_size)))
        return pid, str(dst), h, w

    workers = max(1, cfg.num_workers)
    if workers == 1:
        rows = [_one(pid) for pid in ids]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_one, ids))

    meta = {pid: (path, h, w) for pid, path, h, w in rows if path is not None}
    n_skipped = len(rows) - len(meta)
    if n_skipped:
        logger.warning("%d/%d DICOMs skipped (corrupt/unreadable)", n_skipped, len(rows))

    df = df.copy()
    df = df[df[id_col].isin(meta)]  # drop every row for a skipped id, not just the image
    df["png_path"] = df[id_col].map(lambda p: meta[p][0])
    df["orig_h"] = df[id_col].map(lambda p: meta[p][1])
    df["orig_w"] = df[id_col].map(lambda p: meta[p][2])
    logger.info("cached %d PNGs (%d rows) to %s", len(meta), len(df), out_dir)
    return df
    # ponytail: always decodes DICOM for orig dims even when PNG exists. Add a dims
    # sidecar (json) if re-runs on a warm cache get slow.
