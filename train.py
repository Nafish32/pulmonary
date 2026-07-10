"""CLI entrypoint: python train.py [configs/thesis.yaml]"""

from __future__ import annotations

import sys

from src.config.loader import load_config
from src.pipeline import run_all


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/thesis.yaml"
    cfg = load_config(cfg_path)
    results_md = run_all(cfg)
    print(f"[DONE] results at {results_md}")


if __name__ == "__main__":
    main()
