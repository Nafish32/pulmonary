"""export_split: RSNA pixel boxes -> normalized YOLO labels + data.yaml.

No image decode -- png_path points at dummy files, only label math is checked.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.export import export_split


class _Cfg:  # minimal stand-in; export_split only reads nothing from cfg here
    pass


def _write_dummy_pngs(tmp_path, ids):
    for i in ids:
        (tmp_path / f"{i}.png").write_bytes(b"\x89PNG")


def test_positive_box_normalized(tmp_path):
    _write_dummy_pngs(tmp_path, ["p1"])
    df = pd.DataFrame(
        [dict(patientId="p1", png_path=str(tmp_path / "p1.png"), split="train",
              orig_w=1000, orig_h=1000, x=100, y=200, width=300, height=400, Target=1)]
    )
    out = export_split(df, tmp_path / "ds", _Cfg())
    label = (tmp_path / "ds" / "labels" / "train" / "p1.txt").read_text().strip()
    # xc=(100+150)/1000=.25 yc=(200+200)/1000=.4 w=.3 h=.4
    assert label == "0 0.250000 0.400000 0.300000 0.400000"
    assert "0: opacity" in (tmp_path / "ds" / "data.yaml").read_text()


def test_negative_row_empty_label(tmp_path):
    _write_dummy_pngs(tmp_path, ["p2"])
    df = pd.DataFrame(
        [dict(patientId="p2", png_path=str(tmp_path / "p2.png"), split="val",
              orig_w=800, orig_h=800, x=float("nan"), y=float("nan"),
              width=float("nan"), height=float("nan"), Target=0)]
    )
    export_split(df, tmp_path / "ds", _Cfg())
    assert (tmp_path / "ds" / "labels" / "val" / "p2.txt").read_text() == ""
