"""build_png_cache: a corrupt DICOM must be skipped, not crash the whole cache pass.

Gated like test_smoke_synthetic.py -- needs cv2/pydicom, Kaggle-only in practice.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_HAS_STACK = (
    importlib.util.find_spec("cv2") is not None
    and importlib.util.find_spec("pydicom") is not None
)


class _Cfg:
    def __init__(self, working_root, png_size=32, num_workers=2, cache_png=True):
        self.working_root = str(working_root)
        self.png_size = png_size
        self.num_workers = num_workers
        self.cache_png = cache_png


@pytest.mark.skipif(not _HAS_STACK, reason="needs cv2/pydicom -- runs on Kaggle, not CI")
def test_corrupt_dicom_skipped_not_raised(tmp_path, monkeypatch):
    from src.data import cache as cache_mod

    good_ids = ["p1", "p2"]
    bad_id = "p_bad"
    df = pd.DataFrame(
        [dict(patientId=pid, Target=0) for pid in good_ids + [bad_id]]
    )

    def fake_read_dicom(path):
        pid = path.stem
        if pid == bad_id:
            raise RuntimeError("simulated corrupt DICOM")
        return np.zeros((64, 64), dtype=np.uint8)

    monkeypatch.setattr(cache_mod, "read_dicom", fake_read_dicom)

    out = cache_mod.build_png_cache(df, tmp_path / "dicoms", tmp_path / "png", _Cfg(tmp_path))

    assert set(out["patientId"]) == set(good_ids)  # bad_id's row dropped, no crash
    assert (tmp_path / "png" / "p1.png").exists()
    assert not (tmp_path / "png" / f"{bad_id}.png").exists()
