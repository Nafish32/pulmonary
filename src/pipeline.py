"""Top-level orchestrator.

The Kaggle launcher calls ``run_all(cfg)`` for a full run, or imports individual
stage functions for interactive debugging. No business logic lives in the notebook.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config.schema import Config
from .utils.logger import get_logger
from .utils.paths import ensure_dir

log = get_logger(__name__)


def _scaled_gt(test_df, stem, png_size):
    """GT boxes for one image, scaled from original pixels to png_size square.

    Preds come back in png_size coords (model runs on the cached square PNG), so
    GT must be scaled to the same space or every IoU is wrong.
    """
    from .models.predict import load_gt_boxes_xyxy

    g = load_gt_boxes_xyxy(test_df, stem)
    if len(g) == 0:
        return g
    row = test_df[test_df["patientId"] == stem].iloc[0]
    sx, sy = png_size / float(row["orig_w"]), png_size / float(row["orig_h"])
    return g * np.array([sx, sy, sx, sy])


def run_all(cfg: Config) -> str:
    """Run every stage end to end and return the path to results.md."""
    from .calibration.reliability import brier_score, ece_score
    from .calibration.temperature_scaling import fit_temperature
    from .data.cache import build_png_cache
    from .data.discover import discover_datasets
    from .data.split import patient_split
    from .evaluation.metrics import label_tp_fp, map50
    from .evaluation.plots import reliability_diagram
    from .models.detector import load_detector
    from .models.export import export_split
    from .models.predict import predict_boxes
    from .models.train import train_detector
    from .uncertainty.referral import aurc

    work = ensure_dir(Path(cfg.working_root) / "outputs")

    # --- data ---
    ds = discover_datasets(cfg.input_root)
    df = pd.read_csv(ds["rsna_csv"])
    # subset for a quick probe: debug_mode=50 (plumbing), else max_patients (real signal).
    limit = 50 if cfg.debug_mode else cfg.max_patients
    ids = df["patientId"].unique()
    if limit and limit < len(ids):
        keep = np.random.default_rng(cfg.seed).choice(ids, size=limit, replace=False)
        df = df[df["patientId"].isin(keep)]
        log.info("subset to %d/%d patients (seed=%d)", limit, len(ids), cfg.seed)

    tr, va, te = patient_split(df, cfg.split)
    for part, name in ((tr, "train"), (va, "val"), (te, "test")):
        part["split"] = name
    full = pd.concat([tr, va, te], ignore_index=True)
    full = build_png_cache(full, ds["rsna_images_dir"], work / "png", cfg)
    data_yaml = export_split(full, work / "dataset", cfg)

    # --- detector ---
    model, loaded_name = load_detector(cfg.detector_fallback_chain)
    weights, _ = train_detector(model, data_yaml, cfg)
    log.info("trained %s -> %s", loaded_name, weights)

    # --- evaluation on held-out test ---
    test_df = full[full["split"] == "test"]
    test_imgs = test_df["png_path"].drop_duplicates().tolist()
    preds = predict_boxes(model, test_imgs)
    gts = [_scaled_gt(test_df, Path(p).stem, cfg.png_size) for p in test_imgs]

    mAP = map50(preds, gts)
    conf, correct = label_tp_fp(preds, gts)

    lines = [
        "# Results",
        "",
        "Screening/triage **support** artifact -- NOT a diagnostic system.",
        "",
        f"- detector loaded: `{loaded_name}`",
        f"- fast_mode={cfg.fast_mode} epochs={cfg.epochs} png_size={cfg.png_size}",
        f"- kaggle_dataset_version: {cfg.kaggle_dataset_version}",
        f"- test images: {len(test_imgs)}, predictions: {conf.size}",
        f"- **mAP@50**: {mAP:.4f}",
    ]

    if cfg.calibration_enabled and conf.size:
        ece = ece_score(conf, correct, cfg.n_bins)
        brier = brier_score(conf, correct)
        T = fit_temperature(conf, correct)
        reliability_diagram(conf, correct, cfg.n_bins, str(work / "reliability.png"))
        lines += [
            f"- ECE ({cfg.n_bins} bins): {ece:.4f}",
            f"- Brier: {brier:.4f}",
            f"- temperature T: {T:.3f}",
            "- reliability diagram: `reliability.png`",
        ]

    if cfg.uncertainty_enabled and conf.size:
        lines.append(f"- AURC (risk-coverage): {aurc(conf, correct):.4f}")

    results = work / "results.md"
    results.write_text("\n".join(lines) + "\n")
    log.info("wrote %s", results)
    return str(results)
