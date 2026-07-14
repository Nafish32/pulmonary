"""Detector loading with a real runtime fallback chain.

Each name in the chain is attempted with a real try/except around the constructor
and weight download, and the loaded name logged. Never assumed. ``rtdetr*`` names
load through Ultralytics' RTDETR wrapper (transformer, NMS-free) instead of YOLO;
both expose the same ``.train``/``.predict`` API, so the rest of the pipeline is
detector-agnostic -- the second detector for the generality check costs no new code.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _construct(name: str):
    """Build one detector by name. RT-DETR uses the RTDETR class, else YOLO."""
    if str(name).lower().startswith("rtdetr"):
        from ultralytics import RTDETR  # lazy: only imported for an rtdetr chain

        return RTDETR(name)
    from ultralytics import YOLO

    return YOLO(name)


def load_detector(fallback_chain: list[str]):
    """Load the first detector in the chain that resolves at runtime.

    Returns:
        (model, loaded_name). Raises RuntimeError only if the whole chain fails.
    """
    errors = []
    for name in fallback_chain:
        try:
            model = _construct(name)  # resolves + downloads weights if available
            logger.info("loaded detector: %s", name)
            return model, name
        except Exception as e:  # noqa: BLE001 -- try next in chain, report all if none work
            logger.warning("detector %s unavailable: %s", name, e)
            errors.append(f"{name}: {e}")
    raise RuntimeError("no detector in fallback chain loaded:\n" + "\n".join(errors))


def load_weights(family_hint: str, weights_path: str):
    """Load a *trained* checkpoint through the SAME class it was trained with.

    ``family_hint`` is any string that names the detector family (e.g. the
    ``loaded_name``/``detector_model_name`` from training, like ``"rtdetr-l.pt"``
    or ``"yolo26m.pt (reload)"``) -- only the ``rtdetr`` prefix is checked, same
    rule as ``_construct``. Needed because eval code used to hardcode
    ``YOLO(weights)``, which silently misloads an RT-DETR checkpoint (RTDETR and
    YOLO are different Ultralytics wrapper classes over different architectures).
    """
    if str(family_hint).lower().startswith("rtdetr"):
        from ultralytics import RTDETR

        return RTDETR(weights_path)
    from ultralytics import YOLO

    return YOLO(weights_path)
