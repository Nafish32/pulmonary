"""EigenCAM / EigenGradCAM saliency."""

from __future__ import annotations

from .gradcam import run_cam


def eigencam(model, image, target_layer):
    """Return an EigenCAM saliency map (H, W) in [0, 1] for ``image``."""
    from pytorch_grad_cam import EigenCAM

    return run_cam(EigenCAM, model, image, target_layer)
