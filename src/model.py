"""Load YOLOv8s as a plain nn.Module for direct forward() calls.

Bypasses the ultralytics predict() wrapper entirely — no preprocessing,
no NMS, no autocast. The caller owns the input tensor and interprets raw output.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

WEIGHTS_DEFAULT = "yolov8s.pt"  # ultralytics downloads on first call if absent


def load_yolov8s(
    weights: str | Path = WEIGHTS_DEFAULT,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """Return the underlying DetectionModel (nn.Module) in eval mode.

    Args:
        weights: Path to .pt file or a name ultralytics can auto-download.
        device: Target device string or torch.device.

    Returns:
        nn.Module with frozen BN running stats. Call with torch.no_grad() at the benchmark site.
    """
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    model: nn.Module = yolo.model
    model.eval()
    model = model.to(device)
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
