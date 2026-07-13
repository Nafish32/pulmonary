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


def _build_full_df(cfg: Config, work: Path):
    """Stages 1-4: discover -> read labels -> patient split -> DICOM->PNG cache.

    Returns ``(full, ds)`` -- df (split/png_path/orig_h/orig_w) + the discovery
    dict (RSNA + optional VinDr paths). Deterministic given cfg.seed, so run_all
    and eval_from_weights rebuild the SAME test split.
    """
    from .data.cache import build_png_cache
    from .data.discover import discover_datasets
    from .data.split import patient_split

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
    return full, ds


def _positive_test_rows(test_df, n: int):
    """Up to ``n`` unique positive (Target==1) test images -- the ones with a GT
    box, so energy-in-box / mAP-degradation have something to measure against."""
    pos = test_df[test_df["Target"] == 1].drop_duplicates("patientId")
    return pos.head(n)


def _load_img_gt(test_df, png_path, cfg):
    """(uint8 png_size-square array, first GT box xyxy in png_size frame) or (None, None)."""
    import cv2

    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    g = _scaled_gt(test_df, Path(png_path).stem, cfg.png_size)
    return img, (g[0] if len(g) else None)


def _xai_report(model, test_df, cfg) -> list[str]:
    """XAI validation: mean saliency energy-in-GT-box over positive test images.

    EigenCAM (gradient-free -- the right tool for a detection head; gradcam++ needs a
    scalar class logit a YOLO/RT-DETR head lacks). Energy near 1 = saliency concentrates
    on the lesion; near box_area/image_area = no better than uniform. Target layer is
    NOT VERIFIED across ultralytics versions -> guarded: on any failure, log + report
    skip, never crash the run.
    """
    from .explainability.eigencam import eigencam
    from .explainability.evaluation import saliency_energy_in_box

    pos = _positive_test_rows(test_df, cfg.xai_samples)
    if len(pos) == 0:
        return ["- XAI: no positive test images, skipped"]

    # NOT VERIFIED: layer[-2] (last C3k2 before the Detect head) is the usual CAM
    # target for this backbone. Eyeball one map before trusting the number.
    try:
        target = model.model.model[-2]
    except Exception as e:  # noqa: BLE001
        return [f"- XAI: could not resolve target layer ({e}), skipped"]

    # gradcam++ needs a scalar class logit to backprop; a YOLO detection head has
    # none (its forward is a tuple of box/anchor tensors), so it's dropped. EigenCAM
    # is gradient-free -- the right, robust tool for a detector backbone.
    methods = {"eigencam": eigencam}
    energies = {name: [] for name in methods}
    baselines = []   # box_area/img_area per image = the uniform-saliency null. energy
    first = None     # AT/below this = saliency no better than chance (wrong layer or diffuse).
    for r in pos.itertuples():
        img, box = _load_img_gt(test_df, r.png_path, cfg)
        if img is None or box is None:
            continue
        x1, y1, x2, y2 = box
        baselines.append(max(0.0, (x2 - x1) * (y2 - y1)) / float(img.shape[0] * img.shape[1]))
        for name, fn in methods.items():
            try:
                sal = fn(model, img, target)
                e = saliency_energy_in_box(sal, box)
                if e == e:  # not NaN
                    energies[name].append(e)
                    if first is None:
                        first = (img, box, sal)
            except Exception as ex:  # noqa: BLE001 -- one method/image failing != run failing
                log.warning("XAI %s failed on %s: %s", name, r.patientId, ex)

    base = float(np.mean(baselines)) if baselines else float("nan")
    if first is not None:
        _save_xai_overlay(*first, Path(cfg.working_root) / "outputs" / "xai_example.png")

    lines = [f"- XAI saliency energy-in-box (positives; uniform baseline={base:.3f}, higher=better):"]
    any_ok = False
    for name, es in energies.items():
        if es:
            any_ok = True
            m = float(np.mean(es))
            verdict = ("<= uniform: NOT localizing (suspect target layer, see xai_example.png)"
                       if m <= base * 1.15 else f"{m / base:.2f}x uniform")
            lines.append(f"  - {name}: {m:.3f} (n={len(es)}) -- {verdict}")
    return lines if any_ok else ["- XAI: all methods failed (see logs), skipped"]


def _save_xai_overlay(img, box, sal, out_png) -> None:
    """Dump one saliency heatmap over the image with the GT box, for eyeballing whether
    the map lands on the lesion. Guarded by caller's _guarded; cv2 only, no matplotlib."""
    import cv2

    s = np.clip(np.asarray(sal, float), 0, None)
    s = s / s.max() if s.max() > 0 else s
    if s.shape != img.shape:
        s = cv2.resize(s, (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap((255 * s).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.55, heat, 0.45, 0)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(over, (x1, y1), (x2, y2), (0, 255, 0), 2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), over)


def _robustness_report(model, test_df, cfg) -> list[str]:
    """Corruption sweep on a positive-image subset: worst-case mAP drop + ECE drift."""
    from .robustness.benchmark import run_robustness

    pos = _positive_test_rows(test_df, cfg.robustness_samples)
    imgs, gts = [], []
    for r in pos.itertuples():
        img, _ = _load_img_gt(test_df, r.png_path, cfg)
        g = _scaled_gt(test_df, r.patientId, cfg.png_size)
        if img is not None and len(g):
            imgs.append(img)
            gts.append(g)
    if len(imgs) < 5:
        return [f"- robustness: only {len(imgs)} usable positive images, skipped"]

    res = run_robustness(model, imgs, gts, cfg)
    clean_map, clean_ece = res["clean"]
    # worst = biggest mAP drop across all (kind, severity).
    worst_key, (worst_map, worst_ece) = min(
        ((k, v) for k, v in res.items() if k != "clean"), key=lambda kv: kv[1][0]
    )
    import math

    # worst case can wipe out all conf>=0.05 preds -> ECE undefined (nan). Report it
    # as such instead of "nan (drift +nan)" -- the mAP collapse is the real finding.
    ece_str = ("n/a (no conf>=0.05 preds under corruption)" if math.isnan(worst_ece)
               else f"{worst_ece:.4f} (drift {worst_ece - clean_ece:+.4f})")
    return [
        f"- robustness ({len(imgs)} imgs): clean mAP@50={clean_map:.4f} ECE={clean_ece:.4f}",
        f"  - worst case {worst_key}: mAP@50={worst_map:.4f} "
        f"(drop {clean_map - worst_map:.4f}), ECE={ece_str}",
    ]


def _external_report(model, ds, cfg, work, src_score, src_label) -> list[str]:
    """VinDr cross-domain: triage AUROC + calibration drift, then LABEL-FREE recal.

    ``src_score``/``src_label`` = the in-domain (RSNA) image-level triage scores +
    labels. The recalibration is fit source->target with NO VinDr labels; VinDr labels
    are used only to score it. This is the thesis's method contribution.
    """
    if ds.get("vin_csv") is None or ds.get("vin_images_dir") is None:
        return ["- external (VinDr): dataset not attached, skipped"]
    from .calibration.recalibration import evaluate_recalibration
    from .evaluation.external_validation import build_vindr_eval_set, external_validate

    vin_eval = build_vindr_eval_set(ds["vin_csv"], n_max=1000, seed=cfg.seed)
    res = external_validate(model, vin_eval, ds["vin_images_dir"], cfg, work / "vindr_png")
    if res["n"] == 0:
        return ["- external (VinDr): no images could be cached, skipped"]
    lines = [
        f"- external (VinDr, n={res['n']}, no recalibration): "
        f"triage AUROC={res['auroc']:.4f}, image-level ECE={res['ece']:.4f}",
    ]

    # label-free recalibration RSNA(source) -> VinDr(target): does confidence transfer?
    rec = evaluate_recalibration(src_score, src_label, res["scores"], res["labels"],
                                 cfg.n_bins, coverage=0.20)
    a = next(iter(rec.values()))["actual_risk"]  # same across methods (rank-preserving)
    lines.append(
        f"- external recalibration (label-free; top-20% referral, actual risk={a:.4f}):"
    )
    for m, d in rec.items():
        lines.append(
            f"  - {m}: target ECE={d['ece']:.4f}, expected risk={d['expected_risk']:.4f}, "
            f"calib gap={d['calib_gap']:.4f}"
        )
    return lines


def _ensemble_report(models, test_df, cfg) -> list[str]:
    """Mean per-image confidence spread (std across seeds) = ensemble uncertainty."""
    if not models or len(models) < 2:
        return ["- ensemble: single model, spread not meaningful (set ensemble_seeds "
                "to >=2 seeds for this metric)"]
    from .uncertainty.ensemble import ensemble_uncertainty

    imgs = test_df["png_path"].drop_duplicates().tolist()
    spread = ensemble_uncertainty(models, imgs)
    mean_std = float(np.mean([s["std_conf"] for s in spread]))
    return [f"- ensemble ({len(models)} seeds): mean per-image conf std={mean_std:.4f} "
            "(higher = more disagreement = more uncertain)"]


def _guarded(lines, label, fn, *args):
    """Append fn()'s report lines to ``lines``; on any error, append one note and
    continue -- one broken analysis stage must never sink a 12hr run's results."""
    try:
        lines += fn(*args)
    except Exception as e:  # noqa: BLE001
        log.warning("%s stage failed: %s", label, e)
        lines.append(f"- {label}: FAILED ({e})")


def _evaluate(cfg: Config, weights: str, full, ds, loaded_name: str, work: Path,
              ensemble_models=None) -> str:
    """Stages 7-8: predict on held-out test from ``weights`` -> results.md.

    Detection + calibration + referral, then the trust package (XAI, robustness,
    external VinDr, ensemble) -- each guarded so a failure degrades to a note in
    results.md, never a lost run. Always loads best.pt fresh -- never a stale
    in-memory model (a resume run leaves pipeline's model untrained).
    """
    from .calibration.reliability import brier_score, ece_score
    from .calibration.temperature_scaling import fit_temperature
    from .evaluation.metrics import label_tp_fp, map50
    from .evaluation.plots import reliability_diagram
    from .models.predict import predict_boxes
    from .uncertainty.referral import aurc
    from ultralytics import YOLO

    log.info("[7/8] predict on held-out test (from %s)", weights)
    best_model = YOLO(weights)
    test_df = full[full["split"] == "test"]
    test_imgs = test_df["png_path"].drop_duplicates().tolist()
    preds = predict_boxes(best_model, test_imgs, imgsz=cfg.png_size)
    gts = [_scaled_gt(test_df, Path(p).stem, cfg.png_size) for p in test_imgs]

    # in-domain image-level triage (source for source->target recalibration): per image
    # the max box confidence = screening score, and whether it has any GT opacity box.
    # Same granularity as the VinDr external triage, so calibration transfers apples-to-apples.
    src_img_score = np.array([p["scores"].max() if p["scores"].size else 0.0 for p in preds])
    src_img_label = np.array([1.0 if len(g) else 0.0 for g in gts])

    log.info("[8/8] score: mAP@50 + calibration + uncertainty + trust package")
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

    # --- trust package (each guarded; a failure = a note, not a lost run) ---
    if cfg.xai_enabled:
        log.info("[trust] XAI validation")
        _guarded(lines, "XAI", _xai_report, best_model, test_df, cfg)
    if cfg.robustness_enabled:
        log.info("[trust] robustness sweep")
        _guarded(lines, "robustness", _robustness_report, best_model, test_df, cfg)
    if cfg.external_enabled:
        log.info("[trust] external VinDr validation + label-free recalibration")
        _guarded(lines, "external", _external_report, best_model, ds, cfg, work,
                 src_img_score, src_img_label)
    log.info("[trust] ensemble uncertainty")
    _guarded(lines, "ensemble", _ensemble_report, ensemble_models, test_df, cfg)

    # one-line go/no-go banner (also the first line of results.md).
    def _fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    tag = "PROBE" if cfg.fast_mode else "THESIS"
    summary = (
        f"[{tag}] {loaded_name} | patients~{test_df['patientId'].nunique()} test | "
        f"mAP@50={_fmt(mAP)} ECE={_fmt(ece)} AURC={_fmt(risk)} "
        f"(fast_mode={cfg.fast_mode}, epochs={cfg.epochs})"
    )
    lines.insert(1, f"\n`{summary}`\n")

    results = work / "results.md"
    results.write_text("\n".join(lines) + "\n")
    log.info(summary)  # prints loudly at the end of the run
    log.info("wrote %s", results)
    return str(results)


def eval_from_weights(cfg: Config, weights: str) -> str:
    """Skip training: rebuild the test split + score an existing best.pt.

    For recovering a run whose training finished but whose [7/8]/[8/8] eval was
    wrong (or you just want to re-score different weights). Reuses the exact same
    data prep + scoring as run_all, so numbers are comparable. Re-decodes DICOMs
    for orig dims (~minutes on full RSNA) but skips the ~12hr train. Single-model
    (no ensemble spread -- that needs the multiple trained members run_all makes).
    """
    work = ensure_dir(Path(cfg.working_root) / "outputs")
    log.info("=== eval_from_weights START (weights=%s) ===", weights)
    full, ds = _build_full_df(cfg, work)
    return _evaluate(cfg, weights, full, ds, f"{Path(weights).name} (reload)", work)


def run_all(cfg: Config) -> str:
    """Run every stage end to end and return the path to results.md."""
    from .models.detector import load_detector
    from .models.export import export_split
    from .models.train import _run_name, train_detector

    work = ensure_dir(Path(cfg.working_root) / "outputs")
    log.info("=== run_all START (fast_mode=%s epochs=%s) ===", cfg.fast_mode, cfg.epochs)

    full, ds = _build_full_df(cfg, work)

    log.info("[5/8] export YOLO dataset tree")
    data_yaml = export_split(full, work / "dataset", cfg)

    # --- detector (primary, seed=cfg.seed) ---
    log.info("[6/8] load detector (may download weights -- needs Internet ON)")
    model, loaded_name = load_detector(cfg.detector_fallback_chain)
    log.info("[6/8] loaded %s; training (GPU wakes here)", loaded_name)
    weights, _ = train_detector(model, data_yaml, cfg)
    log.info("[6/8] trained %s -> %s", loaded_name, weights)

    # --- ensemble members (extra seeds; each is a FULL train, off by default) ---
    ensemble_models = None
    extra_seeds = [s for s in cfg.ensemble_seeds if s != cfg.seed]
    if extra_seeds:
        from ultralytics import YOLO

        members = [YOLO(weights)]  # primary counts as seed cfg.seed
        for s in extra_seeds:
            log.info("[6/8] ensemble member seed=%d (another full train)", s)
            m, _n = load_detector(cfg.detector_fallback_chain)
            w, _ = train_detector(m, data_yaml, cfg, seed=s,
                                  run_name=f"{_run_name(cfg)}_seed{s}")
            members.append(YOLO(w))
        ensemble_models = members

    return _evaluate(cfg, weights, full, ds, loaded_name, work, ensemble_models)
