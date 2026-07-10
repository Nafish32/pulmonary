"""YOLO dataset export + box geometry.

``clip_box_xywhn`` fixes the M2 bug: clipping only the box *center* to [0,1] left
edge boxes with ``xc + w/2 > 1``, silently mis-training YOLO. Width/height must be
shrunk after center-clipping.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

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
