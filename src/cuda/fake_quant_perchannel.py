"""Python wrapper for the per-channel INT8 fake-quantize CUDA kernel.

JIT-compiles the extension on first import (cached to ~/.cache/torch_extensions/).
Falls back to a pure-PyTorch reference if CUDA is unavailable (e.g. on Mac).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_ext = None  # loaded lazily

_SRC = Path(__file__).parent / "fake_quant_perchannel.cu"


def _load_extension():
    global _ext
    if _ext is not None:
        return _ext
    # Pin to the target arch to avoid recompiling across processes.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    from torch.utils.cpp_extension import load
    _ext = load(
        name="fake_quant_perchannel",
        sources=[str(_SRC)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _ext


def fake_quant_perchannel(weight: torch.Tensor) -> torch.Tensor:
    """Per-channel symmetric INT8 fake-quantize a weight tensor.

    Each output channel gets its own scale: scale_c = max(|W[c,:]|) / 127.
    The returned tensor has the same shape and dtype as weight but with values
    rounded to the nearest INT8 representable value and dequantized back to float32.

    Args:
        weight: float32 CUDA tensor of shape [C_out, ...] or [C_out, C_in, kH, kW].

    Returns:
        float32 tensor of the same shape.
    """
    if not weight.is_cuda:
        return _fake_quant_perchannel_cpu(weight)

    original_shape = weight.shape
    # Flatten to [C_out, spatial_size] for the kernel.
    C_out = original_shape[0]
    w2d = weight.reshape(C_out, -1).contiguous()

    ext = _load_extension()
    out2d = ext.fake_quant_perchannel(w2d)
    return out2d.reshape(original_shape)


def _fake_quant_perchannel_cpu(weight: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch reference (CPU fallback). Used for correctness tests."""
    original_shape = weight.shape
    C_out = original_shape[0]
    w2d = weight.reshape(C_out, -1)
    abs_max = w2d.abs().max(dim=1).values  # [C_out]
    scales = abs_max / 127.0
    scales = scales.clamp(min=1e-8)        # avoid div-by-zero
    w_scaled = w2d / scales.unsqueeze(1)
    w_q = w_scaled.round_().clamp_(-128, 127).mul_(scales.unsqueeze(1))
    return w_q.reshape(original_shape)


def apply_weight_fake_quant_perchannel(
    model: torch.nn.Module,
    skip_layers: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Apply per-channel fake-quant in-place to all Conv2d layers in model.

    Args:
        model: nn.Module to quantize.
        skip_layers: set of layer name prefixes to skip (e.g. {"model.22.dfl"}).

    Returns:
        Dict mapping layer name → original weight tensor (for restoration).
    """
    from src.quantize import iter_conv_modules

    skip_layers = skip_layers or set()
    saved: dict[str, torch.Tensor] = {}

    for name, conv in iter_conv_modules(model):
        if any(name.startswith(s) for s in skip_layers):
            continue
        saved[name] = conv.weight.data.clone()
        w = conv.weight.data
        if w.is_cuda:
            conv.weight.data = fake_quant_perchannel(w)
        else:
            conv.weight.data = _fake_quant_perchannel_cpu(w)

    return saved
