"""Grad-CAM++ over detector feature maps.

NOT VERIFIED: the exact target layer for a YOLO backbone and how pytorch-grad-cam
hooks Ultralytics' nn.Module wrapper vary by version. Pass an explicit target
layer (e.g. model.model.model[-2]) and smoke-test the map before trusting numbers.
"""

from __future__ import annotations

import numpy as np


def run_cam(cam_cls, model, image, target_layer):
    """Run any pytorch-grad-cam class -> (H, W) saliency in [0, 1].

    Shared by gradcam/eigencam/scorecam. Model-bound (torch + grad-cam) -> Kaggle
    only. Caller supplies the target layer explicitly (backbone/version dependent).
    """
    import torch

    arr = np.asarray(image, np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[None], 3, axis=0)  # 1->3 channel for the CNN
    tensor = torch.from_numpy(arr[None]).float()

    net = getattr(model, "model", model)  # unwrap Ultralytics -> nn.Module
    with cam_cls(model=net, target_layers=[target_layer]) as cam:
        return cam(input_tensor=tensor)[0]  # (H, W), already 0..1


def gradcam(model, image, target_layer):
    """Return a Grad-CAM++ saliency map (H, W) in [0, 1] for ``image``."""
    from pytorch_grad_cam import GradCAMPlusPlus

    return run_cam(GradCAMPlusPlus, model, image, target_layer)
