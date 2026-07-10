"""2-seed ensemble uncertainty.

Guard: assert len(models) >= 2 before computing spread, else label output
'single-model, uncertainty not meaningful' -- never report std_conf=0 as spread.
"""

from __future__ import annotations

import numpy as np


def ensemble_uncertainty(models: list, image_paths):
    """Per-box mean/std confidence across ensemble members.

    Raises if fewer than 2 members (caller decides to label vs skip).
    """
    if len(models) < 2:
        raise ValueError(
            f"ensemble needs >=2 models, got {len(models)}; "
            "label section 'single-model, uncertainty not meaningful' instead"
        )
    from src.models.predict import predict_boxes  # lazy: needs live models

    # per-image triage score (max box conf) from each member -> mean/std spread.
    per_model = [
        [p["scores"].max() if p["scores"].size else 0.0 for p in predict_boxes(m, image_paths)]
        for m in models
    ]
    stack = np.asarray(per_model, float)  # (n_models, n_images)
    return [
        {"mean_conf": float(stack[:, i].mean()), "std_conf": float(stack[:, i].std())}
        for i in range(stack.shape[1])
    ]
