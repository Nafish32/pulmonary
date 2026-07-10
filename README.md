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
pytest tests/ -q                    # config schema, split leakage, box clipping
python train.py configs/debug.yaml  # (once stages are implemented)
```

## Configs

| file | fast_mode | epochs | use |
|------|-----------|--------|-----|
| `configs/thesis.yaml` | false | 50 | full thesis numbers |
| `configs/fast.yaml`   | true  | 5  | smoke the chain on Kaggle |
| `configs/debug.yaml`  | true  | 1  | 5-image synthetic, CPU, <2 min |

Config is pydantic-validated: unknown keys and missing `fast_mode`/`epochs` fail
at load, not 40 minutes into training.

## Status

All stages implemented and wired into `pipeline.run_all`.

**Real + unit-tested (CI, no GPU):** config schema/loader, patient split + leakage
assertion, YOLO export/box-clipping, IoU + greedy TP/FP matching, mAP@50, ECE/Brier,
temperature scaling, risk-coverage/AURC, corruptions (noise/contrast/downsample),
saliency energy-in-box, reliability diagram, VinDr binary filter, AUROC.

**Real but Kaggle-only (need torch/ultralytics/cv2/pydicom + GPU):** DICOM decode,
PNG cache, detector load (fallback chain), training (OOM retry), inference,
Grad-CAM++, deletion curve, robustness sweep, ensemble spread, external validation.
The synthetic 5-image `run_all` smoke test auto-runs where that stack is present,
skips in CI.

**Not yet wired into `run_all`:** robustness sweep, external VinDr eval, 2-seed
ensemble (each needs extra data/array loading or a second trained model).

## Not verified

Exact `ultralytics` version shipping `yolo26m.pt`. Pin the version you smoke-test
(`pip show ultralytics`); the fallback chain (yolo26m → yolo11m → yolov8m) covers
older pins.

Grad-CAM++ target layer for a YOLO backbone and how `pytorch-grad-cam` hooks the
Ultralytics wrapper — pass the target layer explicitly and eyeball a map before
trusting XAI numbers. `model.trainer.best` weights attr (fallback to
`save_dir/weights/best.pt` is coded).
