"""Detector training (mixed precision, OOM retry at fallback_imgsz, checkpoint resume)."""

from __future__ import annotations

from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _run_name(cfg, run_name: str | None = None) -> str:
    """fast.yaml/debug.yaml vs thesis.yaml get separate checkpoint slots.

    Without this, switching configs in the same Kaggle session (e.g. probe
    with fast.yaml, then the real thesis.yaml run) would find the probe's
    last.pt sitting at the shared path and try to resume IT -- wrong
    imgsz/batch/dataset, silently wrong numbers or a hard error.

    ``run_name`` overrides the default slot -- used to give each ensemble seed
    its own checkpoint (e.g. train_thesis_seed123) so members don't collide.
    """
    if run_name is not None:
        return run_name
    return "train_thesis" if not cfg.fast_mode else "train_fast"


def checkpoint_path(cfg, run_name: str | None = None) -> Path:
    """Stable last.pt location, outside the repo clone.

    Kaggle launcher Cell 1 does ``rm -rf`` on the repo dir every re-run (fresh
    clone). A checkpoint saved under the default Ultralytics project dir (cwd-
    relative, i.e. inside the repo clone) would be wiped before a resume could
    ever see it. This path lives under working_root instead, a sibling of the
    repo clone, so it survives a Cell 1 re-run within the same Kaggle session.
    """
    return Path(cfg.working_root) / "outputs" / "runs" / _run_name(cfg, run_name) / "weights" / "last.pt"


def train_detector(model, data_yaml: str, cfg, seed: int | None = None,
                   run_name: str | None = None):
    """Train the detector; retry once at fallback_imgsz on CUDA OOM.

    Resumes from ``checkpoint_path(cfg, run_name)`` if it already exists, instead
    of retraining from scratch -- the fix for a run cut off partway through a long
    thesis-scale epoch count (Kaggle session time limit).

    ``seed`` / ``run_name`` override cfg.seed and the checkpoint slot so an
    ensemble can train several members (distinct seeds, distinct slots) through
    the same code path. Both default to the single-model behavior.

    Returns:
        (best_weights_path, results).
    """
    seed = cfg.seed if seed is None else seed
    slot = _run_name(cfg, run_name)
    last = checkpoint_path(cfg, run_name)
    # ponytail: one fixed run slot per (mode, seed) via exist_ok=True -- no
    # multi-run history. Fine since one run per member is the actual use pattern.
    kw = dict(
        data=data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        seed=seed,
        workers=cfg.num_workers,
        amp=True,  # mixed precision
        verbose=True,
        project=str(last.parent.parent.parent),
        name=slot,
        exist_ok=True,
    )
    if cfg.device:  # "" -> ultralytics auto (GPU 0); "0,1" -> DDP across both
        kw["device"] = cfg.device

    if last.exists():
        from ultralytics import YOLO

        logger.info("resuming from checkpoint %s (skipping retrain from scratch)", last)
        model = YOLO(str(last))
        kw["resume"] = True

    try:
        results = model.train(**kw)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        import torch

        torch.cuda.empty_cache()
        logger.warning(
            "CUDA OOM at imgsz=%d batch=%d -> retry at imgsz=%d batch=%d",
            cfg.imgsz, cfg.batch, cfg.fallback_imgsz, max(1, cfg.batch // 2),
        )
        retry_kw = {**kw, "imgsz": cfg.fallback_imgsz, "batch": max(1, cfg.batch // 2)}
        retry_kw.pop("resume", None)  # resume pins the checkpoint's original imgsz/batch
        results = model.train(**retry_kw)

    # Ultralytics stores best weights at trainer.best (Path). Not verified across
    # every version; fall back to save_dir/weights/best.pt if the attr is absent.
    best = getattr(getattr(model, "trainer", None), "best", None)
    if best is None:
        best = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("training done, best weights: %s", best)
    return str(best), results
