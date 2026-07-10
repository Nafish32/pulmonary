"""DICOM -> array. RescaleSlope/Intercept, MONOCHROME1 inversion, VOI LUT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


def read_dicom(path: str | Path) -> np.ndarray:
    """Read a DICOM into a normalized uint8 grayscale array (H, W).

    Order: modality LUT (RescaleSlope/Intercept) -> VOI LUT/windowing ->
    MONOCHROME1 inversion -> percentile normalize to [0, 255]. Blank frames
    (max == min) return all-zeros instead of dividing by zero.

    Raises:
        RuntimeError: on a corrupt/unreadable file (caller logs and skips).
    """
    try:
        ds = pydicom.dcmread(str(path))
        arr = apply_modality_lut(ds.pixel_array, ds)
        arr = apply_voi_lut(arr, ds)
        arr = arr.astype(np.float32)

        # MONOCHROME1: high value = dark. Invert so bone is bright like MONOCHROME2.
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr

        lo, hi = np.percentile(arr, [1, 99])
        if hi <= lo:  # blank / degenerate frame
            return np.zeros(arr.shape, dtype=np.uint8)
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        return (arr * 255).astype(np.uint8)
    except Exception as e:  # noqa: BLE001 -- decode is I/O + vendor quirks; caller skips
        raise RuntimeError(f"failed to decode DICOM {path}: {e}") from e
