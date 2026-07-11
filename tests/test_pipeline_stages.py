"""Pure orchestration logic of the trust-package stages (no torch/model).

Model-bound bodies (gradcam, run_robustness, external_validate) are Kaggle-only;
here we test the sample-selection, the fail-safe guard, and the VinDr cap.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.external_validation import build_vindr_eval_set
from src.pipeline import _guarded, _positive_test_rows


def test_positive_rows_picks_targets_and_caps():
    df = pd.DataFrame([
        dict(patientId="p1", Target=1),
        dict(patientId="p1", Target=1),  # dup id -> counted once
        dict(patientId="p2", Target=0),  # negative -> excluded
        dict(patientId="p3", Target=1),
        dict(patientId="p4", Target=1),
    ])
    out = _positive_test_rows(df, n=2)
    assert len(out) == 2
    assert set(out["patientId"]) <= {"p1", "p3", "p4"}
    assert (out["Target"] == 1).all()


def test_positive_rows_empty_when_no_positives():
    df = pd.DataFrame([dict(patientId="p1", Target=0)])
    assert len(_positive_test_rows(df, n=5)) == 0


def test_guarded_appends_on_success():
    lines = []
    _guarded(lines, "XAI", lambda: ["- ok line"])
    assert lines == ["- ok line"]


def test_guarded_catches_and_notes_failure():
    lines = ["- keep me"]

    def boom():
        raise RuntimeError("kaboom")

    _guarded(lines, "robustness", boom)
    assert lines[0] == "- keep me"  # earlier results survive
    assert "robustness: FAILED" in lines[1] and "kaboom" in lines[1]


def test_vindr_eval_set_n_max_caps_each_class():
    df = pd.DataFrame(
        [dict(image_id=f"pos{i}", class_name="Lung Opacity") for i in range(50)]
        + [dict(image_id=f"neg{i}", class_name="No finding") for i in range(50)]
    )
    ev = build_vindr_eval_set(df, n_max=10, seed=1)
    assert (ev["label"] == 1).sum() == 5  # n_max//2 per class
    assert (ev["label"] == 0).sum() == 5
