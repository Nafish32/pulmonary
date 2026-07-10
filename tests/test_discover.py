"""discover_datasets: marker-based resolution, required-vs-optional behavior.

Pure filesystem test -- no DICOM decode, so it needs no pydicom/cv2.
"""

from __future__ import annotations

import pytest

from src.data.discover import discover_datasets


def _make_rsna(root):
    (root / "rsna").mkdir(parents=True)
    (root / "rsna" / "stage_2_train_labels.csv").write_text("patientId,x,y,width,height,Target\n")
    imgs = root / "rsna" / "stage_2_train_images"
    imgs.mkdir()
    (imgs / "abc.dcm").write_bytes(b"\x00")
    return imgs


def test_finds_rsna(tmp_path):
    imgs = _make_rsna(tmp_path)
    out = discover_datasets(tmp_path)
    assert out["rsna_csv"].name == "stage_2_train_labels.csv"
    assert out["rsna_images_dir"] == imgs
    assert out["vin_csv"] is None  # optional, absent


def test_missing_rsna_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="RSNA not found"):
        discover_datasets(tmp_path)


def test_empty_image_dir_is_not_rsna(tmp_path):
    # csv present but image dir has no .dcm -> treated as not found
    (tmp_path / "stage_2_train_labels.csv").write_text("x\n")
    (tmp_path / "stage_2_train_images").mkdir()
    with pytest.raises(FileNotFoundError):
        discover_datasets(tmp_path)


def test_finds_optional_vin(tmp_path):
    _make_rsna(tmp_path)
    vin = tmp_path / "vinbigdata-x"
    vin.mkdir()
    (vin / "train.csv").write_text("image_id,class_name\n")
    train = vin / "train"
    train.mkdir()
    (train / "img.dcm").write_bytes(b"\x00")
    out = discover_datasets(tmp_path)
    assert out["vin_csv"].parent == vin
    assert out["vin_images_dir"] == train
