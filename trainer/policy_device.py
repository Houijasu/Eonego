"""Shared PyTorch device selection helpers for trainer scripts."""

import torch


def select_device(name: str) -> torch.device:
    """Return the requested torch device, failing clearly when CUDA is required but unavailable."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested with --device cuda, but torch.cuda.is_available() is false. "
                "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver/GPU."
            )
        return torch.device("cuda")
    raise ValueError(f"unknown device mode: {name}")


def device_summary(device: torch.device) -> str:
    """Return one-line diagnostics for reproducible trainer logs."""
    parts = [
        f"device: {device}",
        f"torch: {torch.__version__}",
        f"torch_cuda: {torch.version.cuda or 'none'}",
        f"cuda_available: {torch.cuda.is_available()}",
    ]
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        parts.append(f"gpu: {torch.cuda.get_device_name(index)}")
    return " | ".join(parts)
