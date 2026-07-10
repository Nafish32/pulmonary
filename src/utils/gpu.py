"""GPU probe."""

from __future__ import annotations


def gpu_info() -> str:
    """Return the CUDA device name, or 'no GPU' if unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "no GPU"
