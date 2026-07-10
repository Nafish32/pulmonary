"""to_cxr_channels: raw stacking (pure numpy); cv2 variants are Kaggle-only."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocessing import to_cxr_channels


def test_raw_triplet_shape():
    gray = np.full((16, 16), 100, np.uint8)
    out = to_cxr_channels(gray, ["raw", "raw", "raw"])
    assert out.shape == (16, 16, 3)
    assert (out[..., 0] == gray).all()


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown variant"):
        to_cxr_channels(np.zeros((4, 4), np.uint8), ["sharpen"])
