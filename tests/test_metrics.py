"""iou_xyxy, label_tp_fp, map50 sanity."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import iou_xyxy, label_tp_fp, map50


def test_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]])
    assert np.isclose(iou_xyxy(a, a)[0, 0], 1.0)
    b = np.array([[100, 100, 110, 110]])
    assert iou_xyxy(a, b)[0, 0] == 0.0


def test_iou_half_overlap():
    a = np.array([[0, 0, 10, 10]])          # area 100
    b = np.array([[5, 0, 15, 10]])          # area 100, inter 50 -> iou 50/150
    assert np.isclose(iou_xyxy(a, b)[0, 0], 50 / 150)


def test_label_tp_fp_one_hit_one_miss():
    preds = [{"boxes": np.array([[0, 0, 10, 10], [50, 50, 60, 60]]),
              "scores": np.array([0.9, 0.8])}]
    gts = [np.array([[0, 0, 10, 10]])]  # only first pred matches
    conf, correct = label_tp_fp(preds, gts)
    assert list(correct) == [1.0, 0.0]
    assert list(conf) == [0.9, 0.8]


def test_map50_perfect_is_one():
    preds = [{"boxes": np.array([[0, 0, 10, 10]]), "scores": np.array([0.9])}]
    gts = [np.array([[0, 0, 10, 10]])]
    assert np.isclose(map50(preds, gts), 1.0)


def test_map50_no_gt_is_nan():
    preds = [{"boxes": np.zeros((0, 4)), "scores": np.zeros((0,))}]
    assert np.isnan(map50(preds, [np.zeros((0, 4))]))
