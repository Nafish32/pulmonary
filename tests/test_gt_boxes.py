"""load_gt_boxes_xyxy: RSNA x,y,w,h -> pixel xyxy; negatives -> empty."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.predict import load_gt_boxes_xyxy


def test_positive_boxes_xyxy():
    df = pd.DataFrame([
        dict(patientId="p1", x=10, y=20, width=30, height=40, Target=1),
        dict(patientId="p1", x=5, y=5, width=10, height=10, Target=1),
    ])
    got = load_gt_boxes_xyxy(df, "p1")
    assert got.shape == (2, 4)
    assert np.allclose(got[0], [10, 20, 40, 60])


def test_negative_is_empty():
    df = pd.DataFrame([
        dict(patientId="p2", x=np.nan, y=np.nan, width=np.nan, height=np.nan, Target=0),
    ])
    assert load_gt_boxes_xyxy(df, "p2").shape == (0, 4)
