"""D-RISE saliency: mask geometry + score-weighting, model mocked (offline)."""

import numpy as np

from src.explainability.drise import _rise_masks, d_rise


def test_masks_shape_and_range():
    rng = np.random.default_rng(0)
    m = _rise_masks(50, 8, 64, 48, rng)
    assert m.shape == (50, 64, 48)
    assert m.min() >= 0.0 and m.max() <= 1.0  # bilinear of {0,1} stays in [0,1]


def test_saliency_tracks_the_region_that_keeps_score_high(monkeypatch):
    # Fake detector: high score only when the image's left half is mostly visible
    # (mask preserved it). D-RISE must then light up the LEFT half, not the right.
    H, W = 32, 32

    def fake_predict(model, imgs, imgsz=640, **kw):
        out = []
        for im in imgs:
            left_kept = im[:, : W // 2].mean() > im[:, W // 2:].mean()
            out.append({"scores": np.array([0.9]) if left_kept else np.zeros((0,)),
                        "boxes": np.zeros((0, 4))})
        return out

    monkeypatch.setattr("src.models.predict.predict_boxes", fake_predict)
    img = np.full((H, W), 200, np.uint8)
    sal = d_rise(None, img, imgsz=32, n_masks=400, grid=8, seed=1)
    assert sal[:, : W // 2].mean() > sal[:, W // 2:].mean()


def test_no_detection_under_any_mask_returns_zero(monkeypatch):
    def blind(model, imgs, imgsz=640, **kw):
        return [{"scores": np.zeros((0,)), "boxes": np.zeros((0, 4))} for _ in imgs]

    monkeypatch.setattr("src.models.predict.predict_boxes", blind)
    sal = d_rise(None, np.zeros((16, 16), np.uint8), imgsz=16, n_masks=20)
    assert not sal.any()
