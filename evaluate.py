"""CLI entrypoint for evaluation-only runs: python evaluate.py [config] [weights]

Skeleton: wire to the eval stages once src/pipeline exposes them.
"""

from __future__ import annotations

import sys

from src.config.loader import load_config


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/thesis.yaml"
    cfg = load_config(cfg_path)
    raise NotImplementedError("skeleton: wire evaluate-only path")


if __name__ == "__main__":
    main()
