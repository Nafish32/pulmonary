# Runbook

How to run the pipeline and how code flows from your laptop to Kaggle.

The split is deliberate: **you edit on the laptop, you run on Kaggle.** Kaggle
has the GPU and the RSNA/VinDr data; your laptop does not. Code travels laptop
→ GitHub → Kaggle. Nothing runs the full model locally.

---

## 0. One-time setup (laptop)

```bash
cd "C:/Users/sahad/Desktop/thesis chest xray/project"

# point local git at the repo (skip if already done)
git remote -v            # if it already shows origin -> Nafish32/pulmonary, skip the next line
git remote add origin https://github.com/Nafish32/pulmonary.git

# install the light deps so tests run locally (no torch/cv2/ultralytics needed)
pip install -e ".[dev]"
```

Local env only needs pydantic / pyyaml / numpy / pandas / matplotlib / pytest.
Heavy deps (torch, ultralytics, cv2, pydicom) are lazy-imported, so tests pass
without them; model-bound tests skip themselves.

---

## 1. Run it on Kaggle (the real run)

1. New Kaggle notebook → **File → Import Notebook** → upload
   `notebooks/kaggle_launcher.ipynb` (or paste its cells).
2. Right panel **Add data** → attach the RSNA dataset
   (must contain `stage_2_train_labels.csv` + `stage_2_train_images/`).
   VinDr is optional.
3. Settings → **Accelerator = GPU**. Off = it will crawl.
4. Run all cells. Order the notebook does for you:
   - **Cell 1** clones the pinned commit + installs. Prints `[GIT] running commit: <sha>`.
   - **Cell 2** loads `configs/fast.yaml` (the probe). Swap to `configs/thesis.yaml` for real numbers.
   - **Cell 3** `run_all(cfg)` — data → train → eval → calibration → uncertainty.
   - **Cell 4** renders `results.md`.
5. Read the top line of the output:
   ```
   [PROBE] yolo26m.pt | patients~225 test | mAP@50=0.3xxx ECE=0.0xxx AURC=0.1xxx (fast_mode=True, epochs=5)
   ```
   That is your go/no-go. `n/a` = that stage was off or had no predictions.

### Which config

| Config | What | When |
|---|---|---|
| `configs/debug.yaml` | 50 patients, 1 epoch, no cache | smoke the plumbing (minutes) |
| `configs/fast.yaml` | 1500-patient slice, 5 epochs | **methodology probe** — is the path worth it |
| `configs/thesis.yaml` | full RSNA, 40 epochs | the numbers you write up |

To change the probe size, edit `max_patients` in `configs/fast.yaml` (bump
toward ~26k as you trust the path; `None` = all).

---

## 2. Make a change (laptop → GitHub → Kaggle)

This is the whole loop. Do it every time you change code.

```bash
# a) edit code on the laptop (src/..., configs/..., etc.)

# b) prove it still works locally
pytest -q

# c) ship it
git add -A
git commit -m "feat: <what changed>"     # message ends with the Co-Authored-By line, see below
git push origin main

# d) on Kaggle: re-run Cell 1. It does `rm -rf` + fresh clone of main,
#    so it picks up your push. Then Run All.
```

Commit messages must end with:

```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

### Getting others' changes back down (pull)

If the repo changed somewhere else (another machine, GitHub web edit):

```bash
git pull origin main      # fast-forward your laptop
```

Kaggle never needs `pull` — Cell 1 always clones fresh, so it is always at the
tip of whatever branch/commit `COMMIT` names.

---

## 3. Pin a commit for thesis numbers

`COMMIT = "main"` (in Cell 1) is right while iterating — always newest code.
But a number you put in the thesis must be reproducible, so pin it:

```bash
# on the laptop, after the commit whose numbers you want to keep
git tag thesis-run-v1
git push origin thesis-run-v1
```

Then in Kaggle Cell 1 set `COMMIT = "thesis-run-v1"`. Now that run is frozen —
re-runnable months later against the exact same code. Cell 1 also prints the
resolved SHA, and `results.md` records `kaggle_dataset_version`, so any number
is traceable to code + data.

---

## 4. Fast checks

```bash
pytest -q                       # whole suite (~50 tests, 2 skip locally)
pytest tests/test_pipeline* -q  # just the wiring, if it exists
python -c "from src.config.loader import load_config; load_config('configs/fast.yaml')"   # config sanity
```

If a config typo slips in, load fails immediately (`extra="forbid"`), not 40
minutes into training.
