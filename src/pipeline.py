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
    log.info("=== run_all START (fast_mode=%s epochs=%s) ===", cfg.fast_mode, cfg.epochs)

    # --- data ---
    log.info("[1/8] discover datasets under %s (rglob -- can be slow)", cfg.input_root)
    ds = discover_datasets(cfg.input_root)
    log.info("[1/8] found rsna_csv=%s images=%s", ds["rsna_csv"], ds["rsna_images_dir"])

    log.info("[2/8] read labels csv")
    df = pd.read_csv(ds["rsna_csv"])
    # subset for a quick probe: debug_mode=50 (plumbing), else max_patients (real signal).
    limit = 50 if cfg.debug_mode else cfg.max_patients
    ids = df["patientId"].unique()
    log.info("[2/8] %d rows, %d patients", len(df), len(ids))
    if limit and limit < len(ids):
        keep = np.random.default_rng(cfg.seed).choice(ids, size=limit, replace=False)
        df = df[df["patientId"].isin(keep)]
        log.info("[2/8] subset to %d/%d patients (seed=%d)", limit, len(ids), cfg.seed)

    log.info("[3/8] patient-wise split")
    tr, va, te = patient_split(df, cfg.split)
    for part, name in ((tr, "train"), (va, "val"), (te, "test")):
        part["split"] = name
    full = pd.concat([tr, va, te], ignore_index=True)

    n_imgs = full["patientId"].nunique()
    log.info("[4/8] cache %d DICOMs -> PNG (IO-bound, near-zero CPU)", n_imgs)
    full = build_png_cache(full, ds["rsna_images_dir"], work / "png", cfg)

    log.info("[5/8] export YOLO dataset tree")
    data_yaml = export_split(full, work / "dataset", cfg)

    # --- detector ---
    log.info("[6/8] load detector (may download weights -- needs Internet ON)")
    model, loaded_name = load_detector(cfg.detector_fallback_chain)
    log.info("[6/8] loaded %s; training (GPU wakes here)", loaded_name)
    weights, _ = train_detector(model, data_yaml, cfg)
    log.info("[6/8] trained %s -> %s", loaded_name, weights)

    # --- evaluation on held-out test ---
    # Load best.pt for eval, NOT the in-memory `model`. On a resume run
    # train_detector rebinds its own local YOLO(last.pt) and trains that;
    # pipeline's `model` stays the untrained pretrained object -> every pred
    # garbage -> mAP=0. Reloading `weights` also gets BEST, not last-epoch.
    from ultralytics import YOLO

    log.info("[7/8] predict on held-out test (from %s)", weights)
    best_model = YOLO(weights)
    test_df = full[full["split"] == "test"]
    test_imgs = test_df["png_path"].drop_duplicates().tolist()
    preds = predict_boxes(best_model, test_imgs, imgsz=cfg.png_size)
    gts = [_scaled_gt(test_df, Path(p).stem, cfg.png_size) for p in test_imgs]

    log.info("[8/8] score: mAP@50 + calibration + uncertainty")
    mAP = map50(preds, gts)
    conf, correct = label_tp_fp(preds, gts)

    # conf=0.001 floor floods trust metrics with low-conf FPs on background images
    # (most images have zero GT -> automatic FP regardless of confidence), which
    # degenerates ECE/AURC. mAP keeps the full unfiltered set for recall; calibration
    # and referral score only preds a triage system would actually surface.
    TRUST_CONF_GATE = 0.05
    gate = conf >= TRUST_CONF_GATE
    conf_t, correct_t = conf[gate], correct[gate]

    lines = [
        "# Results",
        "",
        "Screening/triage **support** artifact -- NOT a diagnostic system.",
        "",
        f"- detector loaded: `{loaded_name}`",
        f"- fast_mode={cfg.fast_mode} epochs={cfg.epochs} png_size={cfg.png_size}",
        f"- kaggle_dataset_version: {cfg.kaggle_dataset_version}",
        f"- test images: {len(test_imgs)}, predictions: {conf.size} "
        f"(conf>={TRUST_CONF_GATE}: {conf_t.size})",
        f"- **mAP@50**: {mAP:.4f}",
    ]

    ece = risk = None
    if cfg.calibration_enabled and conf_t.size:
        ece = ece_score(conf_t, correct_t, cfg.n_bins)
        brier = brier_score(conf_t, correct_t)
        T = fit_temperature(conf_t, correct_t)
        reliability_diagram(conf_t, correct_t, cfg.n_bins, str(work / "reliability.png"))
        lines += [
            f"- ECE ({cfg.n_bins} bins): {ece:.4f}",
            f"- Brier: {brier:.4f}",
            f"- temperature T: {T:.3f}",
            "- reliability diagram: `reliability.png`",
        ]

    if cfg.uncertainty_enabled and conf_t.size:
        risk = aurc(conf_t, correct_t)
        lines.append(f"- AURC (risk-coverage): {risk:.4f}")

    # one-line go/no-go banner (also the first line of results.md).
    def _fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    summary = (
        f"[PROBE] {loaded_name} | patients~{test_df['patientId'].nunique()} test | "
        f"mAP@50={_fmt(mAP)} ECE={_fmt(ece)} AURC={_fmt(risk)} "
        f"(fast_mode={cfg.fast_mode}, epochs={cfg.epochs})"
    )
    lines.insert(1, f"\n`{summary}`\n")

    results = work / "results.md"
    results.write_text("\n".join(lines) + "\n")
    log.info(summary)  # prints loudly at the end of the run
    log.info("wrote %s", results)
    return str(results)
