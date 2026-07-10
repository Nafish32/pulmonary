"""Load and validate a YAML config, announcing fast_mode loudly."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import Config


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a validated :class:`Config`.

    Args:
        path: Path to a ``configs/*.yaml`` file.

    Returns:
        Validated config.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: On any unknown key, wrong type, or missing
            required field (``fast_mode``, ``epochs``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = Config(**raw)
    # Loud banner: a thesis-numbers run must be impossible to launch in fast mode
    # by accident. The notebook layer may print; here we print deliberately.
    print(
        f"[CONFIG] loaded {path} -- fast_mode={cfg.fast_mode}, epochs={cfg.epochs}, "
        f"detector={cfg.detector_model_name}, split="
        f"{cfg.split.train:.2f}/{cfg.split.val:.2f}/{cfg.split.test:.2f}"
    )
    return cfg
