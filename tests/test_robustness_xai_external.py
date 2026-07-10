"""Pure pieces of the eval half: corruptions, energy-in-box, reliability plot,
VinDr filtering, AUROC. Model/torch-bound functions are Kaggle-only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.external_validation import _auroc, build_vindr_eval_set
from src.evaluation.plots import reliability_diagram
from src.explainability.evaluation import saliency_energy_in_box
from src.robustness.corruption import corrupt


def _img():
    return (np.ones((32, 32), np.uint8) * 128)


def test_corrupt_shape_and_dtype_preserved():
    for kind in ("gaussian_noise", "contrast", "downsample"):  # cv2-free subset
        out = corrupt(_img(), kind, 2)
        assert out.shape == (32, 32) and out.dtype == np.uint8


def test_contrast_reduces_variance():
    img = np.random.default_rng(0).integers(0, 255, (32, 32), dtype=np.uint8)
    assert corrupt(img, "contrast", 3).std() < img.std()


def test_corrupt_rejects_bad_severity():
    with pytest.raises(ValueError):
        corrupt(_img(), "blur", 5)


def test_energy_in_box_all_inside():
    s = np.zeros((10, 10))
    s[2:5, 2:5] = 1.0
    assert saliency_energy_in_box(s, [2, 2, 5, 5]) == 1.0


def test_energy_in_box_half():
    s = np.zeros((10, 10))
    s[0:2, :] = 1.0  # top two rows
    assert np.isclose(saliency_energy_in_box(s, [0, 0, 10, 1]), 0.5)


def test_reliability_diagram_writes_file(tmp_path):
    pytest.importorskip("matplotlib")  # CI installs it; local env may not
    out = tmp_path / "rel.png"
    reliability_diagram(np.array([0.1, 0.9]), np.array([0.0, 1.0]), 15, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_vindr_filter_binary():
    df = pd.DataFrame([
        dict(image_id="a", class_name="Lung Opacity"),
        dict(image_id="b", class_name="No finding"),
        dict(image_id="c", class_name="No finding"),
    ])
    ev = build_vindr_eval_set(df)
    assert set(ev.label) == {0, 1}
    assert ev.loc[ev.image_id == "a", "label"].iloc[0] == 1


def test_vindr_filter_empty_raises():
    df = pd.DataFrame([dict(image_id="a", class_name="Cardiomegaly")])
    with pytest.raises(ValueError, match="empty"):
        build_vindr_eval_set(df)


def test_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert _auroc(scores, labels) == 1.0
