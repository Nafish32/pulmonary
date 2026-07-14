"""CLI entrypoint: benchmark N checkpoints on the identical metric suite.

    python compare_checkpoints.py out.md \\
        yolo-old=configs/thesis.yaml:runs/yolo_old/best.pt \\
        yolo-upgraded=configs/thesis.yaml:runs/yolo_new/best.pt \\
        rtdetr=configs/rtdetr.yaml:runs/rtdetr/best.pt

Each ``label=config.yaml:weights.pt`` triple is scored via the SAME
``eval_from_weights`` path every other eval in this repo uses (no separate
scoring logic per model), so numbers are directly comparable. Kaggle-only
(needs torch/ultralytics/GPU + the RSNA/VinDr data) -- run it there, not
locally. Writes one comparison table (csv or md, by ``out`` extension) plus
compute-cost columns (param count, per-image inference latency); it does NOT
invent or estimate training wall-clock -- that must come from the Kaggle run
log for each checkpoint if you want it in the table (paste it into the row
afterward, or extend this script to read a metadata file your training run
writes, once one exists).
"""

from __future__ import annotations

import sys
import time

from src.config.loader import load_config
from src.evaluation.comparison import build_comparison_rows, write_comparison_table
from src.models.detector import load_weights
from src.pipeline import eval_from_weights


def _param_count_and_latency(family_hint: str, weights_path: str, imgsz: int,
                              n_warmup: int = 2, n_timed: int = 8) -> tuple[int, float]:
    """Param count + mean per-image inference latency (ms) on random noise input.

    Random input (not real data) is deliberate here: this measures raw forward-
    pass cost, not accuracy, so it doesn't need the dataset. Real accuracy
    numbers come from eval_from_weights on the actual test split, separately.
    """
    import numpy as np

    from src.models.predict import predict_boxes

    model = load_weights(family_hint, weights_path)
    n_params = sum(p.numel() for p in model.model.parameters())
    dummy = [np.random.default_rng(0).integers(0, 255, (imgsz, imgsz), dtype=np.uint8)
             for _ in range(n_warmup + n_timed)]
    for img in dummy[:n_warmup]:
        predict_boxes(model, [img], imgsz=imgsz)
    t0 = time.perf_counter()
    for img in dummy[n_warmup:]:
        predict_boxes(model, [img], imgsz=imgsz)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return n_params, elapsed_ms / n_timed


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: python compare_checkpoints.py <out.md|out.csv> "
                  "label=config.yaml:weights.pt [label2=config2.yaml:weights2.pt ...]")
    out_path = sys.argv[1]
    entries = []
    for spec in sys.argv[2:]:
        label, rest = spec.split("=", 1)
        cfg_path, weights_path = rest.split(":", 1)
        cfg = load_config(cfg_path)

        print(f"[{label}] param count + inference latency (random-input probe)...")
        n_params, latency_ms = _param_count_and_latency(
            cfg.detector_model_name, weights_path, cfg.png_size)

        print(f"[{label}] eval_from_weights (real test split, full metric suite)...")
        results_md_path = eval_from_weights(cfg, weights_path)
        text = open(results_md_path).read()

        entries.append({
            "label": label,
            "results_md_text": text,
            "param_count": n_params,
            "latency_ms_per_img": round(latency_ms, 2),
            "train_hours": None,  # NOT tracked by the pipeline yet -- fill in manually
        })

    rows = build_comparison_rows(entries)
    write_comparison_table(rows, out_path)
    print(f"[DONE] comparison table -> {out_path}")


if __name__ == "__main__":
    main()
