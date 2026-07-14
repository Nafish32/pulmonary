"""YOLO dataset export + box geometry.

``clip_box_xywhn`` fixes the M2 bug: clipping only the box *center* to [0,1] left
edge boxes with ``xc + w/2 > 1``, silently mis-training YOLO. Width/height must be
shrunk after center-clipping.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from src.utils.logger import get_logger
from src.utils.paths import ensure_dir

logger = get_logger(__name__)


def clip_box_xywhn(
    xc: float, yc: float, w: float, h: float
) -> tuple[float, float, float, float]:
    """Clip a normalized YOLO box so it lies fully within [0, 1].

    Center is clamped to [0, 1], then width/height are shrunk so that
    ``xc ± w/2`` and ``yc ± h/2`` stay within [0, 1].

    Returns:
        Corrected ``(xc, yc, w, h)``, all in [0, 1] with w, h >= 0.
    """
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    w = min(w, 2.0 * xc, 2.0 * (1.0 - xc))
    h = min(h, 2.0 * yc, 2.0 * (1.0 - yc))
    return xc, yc, max(w, 0.0), max(h, 0.0)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink src->dst (cheap, no disk); copy if symlinks are unsupported."""
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy(src, dst)


def _augment_train_with_corruptions(root: Path, cfg) -> None:
    """Duplicate a bounded fraction of TRAIN images as corrupted copies (+ same
    label file) so training sees some of what robustness._robustness_report
    tests at eval time -- current baseline has ZERO corruption exposure during
    training, which is exactly why clean mAP@50=0.4563 collapses to 0.0057
    under gaussian_noise severity 3 (see CLAUDE.md robustness_status).

    Deliberately offline (extra files on disk), not a hook into Ultralytics'
    internal augmentation pipeline: trainer-batch tensor shape/dtype/timing is
    version-specific and unverifiable without a live torch/ultralytics install
    in this environment, so hooking it blind would be exactly the kind of
    invented-API risk this repo's rules forbid. Reuses the same, already-tested
    ``corrupt()`` the robustness sweep uses -- same corruption vocabulary
    end to end. Severity 3 is excluded (train on 1-2 only; the most extreme
    severity is kept eval-only to avoid dragging down clean accuracy).

    Guarded per-image: any single failure is logged and skipped, never aborts
    export. Off by default behavior comes from cfg -- getattr with a safe
    default so callers passing a minimal cfg stand-in (as several existing
    tests do) are unaffected.
    """
    kinds = getattr(cfg, "train_corruption_kinds", [])
    severities = getattr(cfg, "train_corruption_severities", [])
    frac = getattr(cfg, "train_corruption_frac", 0.0)
    if not (getattr(cfg, "train_corruption_aug_enabled", False) and kinds and severities and frac > 0):
        return

    import cv2  # lazy: only needed when this augmentation actually runs

    from src.robustness.corruption import corrupt

    train_dir = root / "images" / "train"
    train_imgs = sorted(p for p in train_dir.glob("*.png"))
    n = int(len(train_imgs) * frac)
    if n <= 0:
        return

    rng = np.random.default_rng(getattr(cfg, "seed", 42))
    chosen = rng.choice(np.array(train_imgs, dtype=object), size=n, replace=False)
    made = 0
    for png in chosen:
        try:
            img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            kind = str(rng.choice(kinds))
            severity = int(rng.choice(severities))
            corrupted = corrupt(img, kind, severity)

            stem = png.stem
            suffix = f"_corr_{kind}{severity}"
            out_png = png.parent / f"{stem}{suffix}.png"
            cv2.imwrite(str(out_png), corrupted)

            src_label = root / "labels" / "train" / f"{stem}.txt"
            dst_label = root / "labels" / "train" / f"{stem}{suffix}.txt"
            dst_label.write_text(src_label.read_text() if src_label.exists() else "")
            made += 1
        except Exception as e:  # noqa: BLE001 -- one bad image must not break export
            logger.warning("train corruption-aug skipped for %s: %s", png, e)
    logger.info("train corruption-aug: added %d/%d corrupted train images (kinds=%s, sev=%s)",
                made, n, kinds, severities)


def export_split(df, out_dir, cfg, class_name: str = "opacity") -> str:
    """Write an Ultralytics dataset (images/ + labels/ + data.yaml) from ``df``.

    ``df`` must carry: ``split`` (train/val/test), ``png_path``, ``orig_w``,
    ``orig_h`` (from the PNG cache), and RSNA box cols ``x,y,width,height,Target``.
    RSNA pixel boxes are converted to normalized YOLO xywh on the *original* image
    dims, then ``clip_box_xywhn``'d. Target==0 / NaN rows produce an empty label
    file (YOLO background). Single class 0 = ``class_name``.

    Returns:
        Path (str) to the written data.yaml.
    """
    root = ensure_dir(out_dir)
    for split in ("train", "val", "test"):
        ensure_dir(root / "images" / split)
        ensure_dir(root / "labels" / split)

    # group by image: one PNG has many box rows in RSNA.
    for pid, rows in df.groupby(df["png_path"]):
        split = rows["split"].iloc[0]
        png = Path(pid)
        _link_or_copy(png, root / "images" / split / png.name)

        lines = []
        for r in rows.itertuples():
            if int(getattr(r, "Target", 0)) != 1 or r.width != r.width:  # NaN check
                continue
            ow, oh = float(r.orig_w), float(r.orig_h)
            xc = (float(r.x) + float(r.width) / 2) / ow
            yc = (float(r.y) + float(r.height) / 2) / oh
            xc, yc, w, h = clip_box_xywhn(xc, yc, float(r.width) / ow, float(r.height) / oh)
            if w > 0 and h > 0:
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        label_path = root / "labels" / split / f"{png.stem}.txt"
        label_path.write_text("\n".join(lines))

    _augment_train_with_corruptions(root, cfg)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n  0: {class_name}\n"
    )
    logger.info("exported YOLO dataset -> %s", data_yaml)
    return str(data_yaml)
