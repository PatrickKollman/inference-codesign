"""INT8 quantization utilities for sensitivity analysis and PTQ.

Design note on scope
--------------------
torch.ao.quantization static PTQ targets CPU execution and does not use INT8
tensor cores on CUDA. GPU INT8 latency (the actual speedup story) requires
TensorRT or torch.compile. These utilities support:

  1. Per-layer weight sensitivity: fake-quantize one Conv at a time, measure
     mAP drop. Identifies which layers are precision-sensitive.
  2. Full-model fake-quantized accuracy baseline: all conv weights in INT8
     range, FP32 compute. Measures accuracy loss from weight quantization.

The two outputs (sensitivity ranking + accuracy baseline) feed the Pareto
analysis. The INT8 latency axis of the Pareto is measured via TRT deployment.
"""
from __future__ import annotations

from typing import Generator

import torch
import torch.nn as nn


def fake_quantize_int8_symmetric(x: torch.Tensor) -> torch.Tensor:
    """Simulate per-tensor symmetric INT8 weight quantization.

    Computes the scale as max(|x|) / 127, quantizes to [-128, 127],
    then dequantizes back to float. The result has INT8 precision but
    FP32 dtype — activating the quantization noise without changing the
    compute path.

    Per-tensor (not per-channel) is the conservative baseline. Per-channel
    quantization (one scale per output filter) recovers 1-3 mAP points on
    sensitive layers; we use per-tensor here to expose worst-case sensitivity.
    """
    if x.numel() == 0:
        return x
    scale = x.abs().max().clamp(min=1e-8) / 127.0
    x_q = torch.clamp(torch.round(x / scale), -128, 127)
    return (x_q * scale).to(x.dtype)


def iter_conv_modules(
    model: nn.Module,
) -> Generator[tuple[str, nn.Conv2d], None, None]:
    """Yield (fully-qualified name, module) for every Conv2d in model."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            yield name, module


def count_conv_params(module: nn.Conv2d) -> int:
    """Total weight parameter count for a Conv2d."""
    return module.weight.numel()


def apply_weight_fake_quant(model: nn.Module) -> dict[str, torch.Tensor]:
    """Fake-quantize all Conv2d weights in-place.

    Returns a dict of original weights keyed by module name so the caller
    can restore them with restore_weights().
    """
    originals: dict[str, torch.Tensor] = {}
    for name, conv in iter_conv_modules(model):
        originals[name] = conv.weight.data.clone()
        conv.weight.data = fake_quantize_int8_symmetric(conv.weight.data)
    return originals


def restore_weights(model: nn.Module, originals: dict[str, torch.Tensor]) -> None:
    """Restore Conv2d weights from a dict produced by apply_weight_fake_quant."""
    for name, conv in iter_conv_modules(model):
        if name in originals:
            conv.weight.data = originals[name]
