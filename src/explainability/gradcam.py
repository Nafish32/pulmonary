"""Grad-CAM++ over detector feature maps.

NOT VERIFIED: how pytorch-grad-cam hooks Ultralytics' nn.Module wrapper varies
by version -- smoke-test the map (see pipeline._xai_report's xai_example.png)
before trusting numbers.
"""

from __future__ import annotations

import numpy as np


def resolve_target_layer(model):
    """Find the CAM target layer by introspection instead of a hardcoded index.

    Previous code guessed ``model.model.model[-2]`` (the second-to-last top-level
    block) -- fragile across Ultralytics versions/backbones, and unverified. This
    walks ``model.model.model`` (the nn.Sequential of parsed blocks, true for both
    YOLO's DetectionModel and RT-DETR's RTDETRDetectionModel under Ultralytics'
    shared BaseModel) from the end, skips the final block (the detection/decode
    head -- rarely has a useful spatial conv activation for CAM), and returns the
    last block that actually contains a Conv2d. That's the standard "last
    convolutional block before the head" CAM target, derived from the real
    model structure rather than assumed.

    Still not proven correct for every backbone -- eyeball xai_example.png
    (pipeline._save_xai_overlay) on a real Kaggle run before trusting the
    energy-in-box number. Raises RuntimeError (caller must guard) if no
    Conv2d-bearing block is found at all.
    """
    import torch.nn as nn

    net = getattr(model, "model", model)  # unwrap Ultralytics YOLO/RTDETR wrapper
    seq = getattr(net, "model", None)  # the nn.Sequential of parsed blocks
    try:
        blocks = list(seq) if seq is not None else []
    except TypeError:
        blocks = []
    if not blocks:
        raise RuntimeError("could not find model.model.model (nn.Sequential of blocks)")

    search_order = list(reversed(blocks[:-1])) if len(blocks) > 1 else blocks
    for block in search_order:
        if any(isinstance(m, nn.Conv2d) for m in block.modules()):
            return block
    raise RuntimeError("no Conv2d-bearing block found in model.model.model")


def run_cam(cam_cls, model, image, target_layer):
    """Run any pytorch-grad-cam class -> (H, W) saliency in [0, 1].

    Used by eigencam (the active method). Model-bound (torch + grad-cam) -> Kaggle
    only. Caller supplies the target layer explicitly (backbone/version dependent).
    """
    import torch
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    arr = np.asarray(image, np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[None], 3, axis=0)  # 1->3 channel for the CNN
    tensor = torch.from_numpy(arr[None]).float()

    net = getattr(model, "model", model).float().eval()  # unwrap Ultralytics -> nn.Module

    # Ultralytics forward returns a (preds, feats) TUPLE, not a class-logit tensor.
    # grad-cam's default target path does argmax(outputs.cpu()) on it ->
    # 'tuple' object has no attribute 'cpu'. Passing an explicit target skips that
    # path. EigenCAM is gradient-free (SVD of target-layer activations) and ignores
    # the target's value, so a dummy target works and no backward is run.
    with cam_cls(model=net, target_layers=[target_layer]) as cam:
        return cam(input_tensor=tensor, targets=[ClassifierOutputTarget(0)])[0]  # (H, W) 0..1


def gradcam(model, image, target_layer):
    """Return a Grad-CAM++ saliency map (H, W) in [0, 1] for ``image``."""
    from pytorch_grad_cam import GradCAMPlusPlus

    return run_cam(GradCAMPlusPlus, model, image, target_layer)
