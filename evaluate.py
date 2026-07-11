"""CLI entrypoint for evaluation-only runs: python evaluate.py <config> <weights>

Skips training: rebuilds the test split and scores an existing best.pt via
pipeline.eval_from_weights (same data prep + scoring as a full run).
"""

from __future__ import annotations

import sys

from src.config.loader import load_config
from src.pipeline import eval_from_weights


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: python evaluate.py <config.yaml> <weights.pt>")
    cfg = load_config(sys.argv[1])
    results_md = eval_from_weights(cfg, sys.argv[2])
    print(f"[DONE] results at {results_md}")


if __name__ == "__main__":
    main()
