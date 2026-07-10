"""Detector training (mixed precision, OOM retry at fallback_imgsz)."""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_detector(model, data_yaml: str, cfg):
    """Train the detector; retry once at fallback_imgsz on CUDA OOM.

    Returns:
        (best_weights_path, results).
    """
    kw = dict(
        data=data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        seed=cfg.seed,
        workers=cfg.num_workers,
        amp=True,  # mixed precision
        verbose=True,
    )
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
        results = model.train(**{**kw, "imgsz": cfg.fallback_imgsz, "batch": max(1, cfg.batch // 2)})

    # Ultralytics stores best weights at trainer.best (Path). Not verified across
    # every version; fall back to save_dir/weights/best.pt if the attr is absent.
    best = getattr(getattr(model, "trainer", None), "best", None)
    if best is None:
        from pathlib import Path

        best = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("training done, best weights: %s", best)
    return str(best), results
