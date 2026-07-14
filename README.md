# Trustworthy Pulmonary Opacity Localization (RSNA → VinDr)

Screening/triage **support** research artifact — **not** a diagnostic system.
Novelty is in the evaluation package (calibration + referral + XAI validation +
robustness + cross-domain), not the detector backbone.

## Workflow

Local = source of truth. GitHub = sync. Kaggle = disposable compute that pulls a
**pinned commit** and runs a thin launcher. Never hand-edit code in the Kaggle UI.

```
git tag thesis-run-v1        # tag every numbers-run
# set COMMIT in notebooks/kaggle_launcher.ipynb -> run on Kaggle
```

## Run locally

```bash
pip install -e .[dev]
pytest tests/ -q                    # config schema, split leakage, box clipping, guards
python train.py  configs/debug.yaml            # full run (needs the GPU stack)
python evaluate.py configs/thesis.yaml best.pt # eval-only: score existing weights, skip training
```

## Configs

| file | fast_mode | epochs | patients | use |
|------|-----------|--------|----------|-----|
| `configs/thesis.yaml` | false | 50 | all (~26k) | full thesis numbers |
| `configs/fast.yaml`   | true  | 20 | 5000       | quick-but-real probe on Kaggle |
| `configs/debug.yaml`  | true  | 1  | 50 / 5-img synthetic | CPU smoke, <2 min |

Config is pydantic-validated: unknown keys and missing `fast_mode`/`epochs` fail
at load, not 40 minutes into training.

## Status

All stages implemented and wired into `pipeline.run_all` — detection, calibration,
referral, XAI validation, robustness sweep, external VinDr eval, ensemble spread.
Each trust stage is **guarded**: a failure degrades to a note in `results.md`,
never a lost run. `eval_from_weights` re-scores an existing `best.pt` without
retraining.

**Real + unit-tested (CI, no GPU):** config schema/loader, patient split + leakage
assertion, YOLO export/box-clipping, IoU + greedy TP/FP matching, mAP@50, ECE/Brier,
temperature scaling, risk-coverage/AURC, corruptions (noise/contrast/downsample),
saliency energy-in-box, reliability diagram, VinDr binary filter + cap, AUROC,
corrupt-DICOM skip, checkpoint-slot separation, stage guards.

**Real but Kaggle-only (need torch/ultralytics/cv2/pydicom + GPU):** DICOM decode,
PNG cache, detector load (fallback chain), training (OOM retry + checkpoint resume),
inference, Grad-CAM++/EigenCAM, deletion curve, robustness sweep, ensemble spread,
external validation. The synthetic 5-image `run_all` smoke test auto-runs where that
stack is present, skips in CI.

**Cost knobs (default off/bounded to keep a run tractable):** `ensemble_seeds`
(each extra seed = a full ~12hr train; default single-model), `robustness_samples`
(200-image subset; full set = 15× inference), `xai_samples` (20 positives),
external VinDr caps to 1000 images.

## Not verified

Exact `ultralytics` version shipping `yolo26m.pt`. Pin the version you smoke-test
(`pip show ultralytics`); the fallback chain (yolo26m → yolo11m) covers
older pins.

Grad-CAM++ target layer for a YOLO backbone and how `pytorch-grad-cam` hooks the
Ultralytics wrapper. The XAI stage uses `model.model.model[-2]` (last block before
the Detect head) and is guarded — on failure it writes `XAI: ... skipped` to
`results.md` rather than crashing. **Eyeball one saliency map before trusting the
energy-in-box number.** `model.trainer.best` weights attr (fallback to
`save_dir/weights/best.pt` is coded).

VinDr mirror image format (DICOM vs JPG/PNG) and `class_name` spelling vary; the
external stage tries `.dicom/.dcm/.png/.jpg`, lowercase-normalizes labels, and
prints the unique `class_name` values once. Zero-row filter or no cached images →
stage skips with a note.
