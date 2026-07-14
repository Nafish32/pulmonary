"""export_split: RSNA pixel boxes -> normalized YOLO labels + data.yaml.

No image decode -- png_path points at dummy files, only label math is checked.
"""

from __future__ import annotations

import numpy as np
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


def test_corruption_aug_disabled_by_default_stub_cfg(tmp_path):
    """_Cfg() (no train_corruption_* attrs) must behave exactly like before --
    getattr defaults keep old callers/tests unaffected."""
    _write_dummy_pngs(tmp_path, ["p1"])
    df = pd.DataFrame(
        [dict(patientId="p1", png_path=str(tmp_path / "p1.png"), split="train",
              orig_w=1000, orig_h=1000, x=100, y=200, width=300, height=400, Target=1)]
    )
    export_split(df, tmp_path / "ds", _Cfg())
    imgs = sorted((tmp_path / "ds" / "images" / "train").glob("*.png"))
    assert [p.name for p in imgs] == ["p1.png"]  # no extra corrupted copies


class _CorrCfg:
    seed = 0
    train_corruption_aug_enabled = True
    train_corruption_kinds = ["gaussian_noise"]
    train_corruption_severities = [1]
    train_corruption_frac = 1.0  # corrupt every train image, deterministic to check


def _write_real_png(path, value=128):
    import cv2

    cv2.imwrite(str(path), (np.ones((32, 32), np.uint8) * value))


def test_corruption_aug_adds_train_copy_with_same_label(tmp_path):
    _write_real_png(tmp_path / "p1.png")
    df = pd.DataFrame(
        [dict(patientId="p1", png_path=str(tmp_path / "p1.png"), split="train",
              orig_w=1000, orig_h=1000, x=100, y=200, width=300, height=400, Target=1)]
    )
    export_split(df, tmp_path / "ds", _CorrCfg())

    train_imgs = sorted((tmp_path / "ds" / "images" / "train").glob("*.png"))
    assert len(train_imgs) == 2  # original + one corrupted copy
    corrupted = [p for p in train_imgs if p.name != "p1.png"]
    assert len(corrupted) == 1
    assert "_corr_gaussian_noise1" in corrupted[0].stem

    orig_label = (tmp_path / "ds" / "labels" / "train" / "p1.txt").read_text()
    corr_label_path = tmp_path / "ds" / "labels" / "train" / f"{corrupted[0].stem}.txt"
    assert corr_label_path.read_text() == orig_label  # corruption doesn't move boxes


def test_corruption_aug_does_not_touch_val_or_test(tmp_path):
    _write_real_png(tmp_path / "p1.png")
    _write_real_png(tmp_path / "p2.png")
    df = pd.DataFrame([
        dict(patientId="p1", png_path=str(tmp_path / "p1.png"), split="train",
             orig_w=1000, orig_h=1000, x=100, y=200, width=300, height=400, Target=1),
        dict(patientId="p2", png_path=str(tmp_path / "p2.png"), split="val",
             orig_w=1000, orig_h=1000, x=100, y=200, width=300, height=400, Target=1),
    ])
    export_split(df, tmp_path / "ds", _CorrCfg())
    val_imgs = sorted((tmp_path / "ds" / "images" / "val").glob("*.png"))
    assert [p.name for p in val_imgs] == ["p2.png"]  # untouched


def test_corruption_aug_frac_zero_adds_nothing(tmp_path):
    class _Cfg0(_CorrCfg):
        train_corruption_frac = 0.0

    _write_real_png(tmp_path / "p1.png")
    df = pd.DataFrame(
        [dict(patientId="p1", png_path=str(tmp_path / "p1.png"), split="train",
              orig_w=1000, orig_h=1000, x=100, y=200, width=300, height=400, Target=1)]
    )
    export_split(df, tmp_path / "ds", _Cfg0())
    train_imgs = sorted((tmp_path / "ds" / "images" / "train").glob("*.png"))
    assert [p.name for p in train_imgs] == ["p1.png"]
