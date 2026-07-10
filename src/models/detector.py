"""Detector loading with a real runtime fallback chain.

yolo26m -> yolo11m -> yolov8m, each attempted with a real try/except around
YOLO(name) and the weight download, and the loaded name logged. Never assumed.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_detector(fallback_chain: list[str]):
    """Load the first detector in the chain that resolves at runtime.

    Returns:
        (model, loaded_name). Raises RuntimeError only if the whole chain fails.
    """
    from ultralytics import YOLO  # lazy: keeps module importable without ultralytics

    errors = []
    for name in fallback_chain:
        try:
            model = YOLO(name)  # resolves + downloads weights if available
            logger.info("loaded detector: %s", name)
            return model, name
        except Exception as e:  # noqa: BLE001 -- try next in chain, report all if none work
            logger.warning("detector %s unavailable: %s", name, e)
            errors.append(f"{name}: {e}")
    raise RuntimeError("no detector in fallback chain loaded:\n" + "\n".join(errors))
