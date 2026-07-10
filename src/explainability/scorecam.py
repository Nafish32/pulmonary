"""ScoreCAM saliency (complementary, gradient-free)."""

from __future__ import annotations

from .gradcam import run_cam


def scorecam(model, image, target_layer):
    """Return a ScoreCAM saliency map (H, W) in [0, 1] for ``image``."""
    from pytorch_grad_cam import ScoreCAM

    return run_cam(ScoreCAM, model, image, target_layer)
