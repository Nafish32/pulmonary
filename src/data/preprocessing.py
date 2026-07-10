"""Image preprocessing: percentile norm, optional CLAHE/unsharp channels.

CXR 3-channel enhancement (raw / CLAHE / unsharp) is an input-representation
choice, not a clinical-superiority claim. Compute on the fly per experiment --
do NOT pre-cache the triplet (disk-quota footgun).
"""

from __future__ import annotations

import numpy as np


def to_cxr_channels(gray: np.ndarray, variants: list[str]) -> np.ndarray:
    """Stack requested single-channel variants into H x W x len(variants).

    variants: any of 'raw', 'clahe', 'unsharp'. Computed on the fly (never cached).
    'raw' is pure-numpy; 'clahe'/'unsharp' lazy-import cv2 (Kaggle).
    """
    gray = gray.astype(np.uint8)
    chans = []
    for v in variants:
        if v == "raw":
            chans.append(gray)
        elif v == "clahe":
            import cv2

            chans.append(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray))
        elif v == "unsharp":
            import cv2

            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
            chans.append(cv2.addWeighted(gray, 1.5, blur, -0.5, 0))
        else:
            raise ValueError(f"unknown variant {v!r}; pick from raw/clahe/unsharp")
    return np.stack(chans, axis=-1)
