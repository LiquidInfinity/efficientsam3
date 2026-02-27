"""Device utilities for cross-platform support (CUDA, MPS, CPU)."""

import torch


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_device_type(device) -> str:
    """Return a device type string suitable for ``torch.autocast``.

    MPS does not fully support autocast, so we fall back to ``"cpu"`` for
    non-CUDA devices.
    """
    device_type = device.type if isinstance(device, torch.device) else device
    if device_type == "cuda":
        return "cuda"
    return "cpu"
