"""Image corruptions x severities: gaussian noise, blur, JPEG, contrast, downsample."""

from __future__ import annotations

import numpy as np

CORRUPTIONS = ("gaussian_noise", "blur", "jpeg", "contrast", "downsample")


def corrupt(image: np.ndarray, kind: str, severity: int) -> np.ndarray:
    """Apply corruption ``kind`` at ``severity`` (1-3) to a uint8 grayscale image.

    Deterministic (noise is seeded by severity) so a run is reproducible. Returns
    a uint8 array of the same shape.
    """
    if not 1 <= severity <= 3:
        raise ValueError(f"severity must be 1-3, got {severity}")
    if kind not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {kind!r}; pick from {CORRUPTIONS}")
    img = image.astype(np.float32)
    h, w = img.shape[:2]

    if kind == "gaussian_noise":
        sigma = [8, 16, 32][severity - 1]
        img = img + np.random.default_rng(severity).normal(0, sigma, img.shape)
    elif kind == "contrast":
        factor = [0.75, 0.5, 0.3][severity - 1]
        img = (img - img.mean()) * factor + img.mean()
    elif kind == "downsample":
        f = [2, 4, 8][severity - 1]
        small = img[::f, ::f]
        img = np.repeat(np.repeat(small, f, 0), f, 1)[:h, :w]
    elif kind == "blur":
        import cv2  # lazy: only blur/jpeg need cv2

        k = [3, 5, 7][severity - 1]
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif kind == "jpeg":
        import cv2

        q = [50, 30, 10][severity - 1]
        ok, enc = cv2.imencode(".jpg", img.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)
