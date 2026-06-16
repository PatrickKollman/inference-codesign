"""Correctness tests for the per-channel INT8 fake-quantize kernel.

Tests run on CPU using the pure-PyTorch reference implementation so no GPU
is required for CI. GPU tests are marked and skipped when CUDA is unavailable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.cuda.fake_quant_perchannel import (
    _fake_quant_perchannel_cpu as ref,
    fake_quant_perchannel,
    apply_weight_fake_quant_perchannel,
)
from src.quantize import iter_conv_modules, restore_weights


# ── Reference correctness ────────────────────────────────────────────────────

class TestReferenceCorrectness:
    """Verify the CPU reference implementation semantics."""

    def test_output_dtype_preserved(self):
        x = torch.randn(8, 2304)
        assert ref(x).dtype == torch.float32

    def test_output_shape_preserved(self):
        x = torch.randn(16, 576)
        assert ref(x).shape == x.shape

    def test_4d_input_shape_preserved(self):
        x = torch.randn(32, 16, 3, 3)
        assert ref(x).shape == x.shape

    def test_values_within_perchannel_int8_range(self):
        torch.manual_seed(0)
        x = torch.randn(8, 256)
        out = ref(x)
        for c in range(x.size(0)):
            scale_c = x[c].abs().max() / 127.0
            assert out[c].abs().max() <= 127.0 * scale_c + 1e-5

    def test_unique_values_bounded_by_int8(self):
        x = torch.linspace(-2.0, 2.0, 2560).reshape(10, 256)
        out = ref(x)
        for c in range(10):
            assert out[c].unique().numel() <= 256

    def test_zero_channel_passthrough(self):
        x = torch.randn(4, 64)
        x[2, :] = 0.0
        out = ref(x)
        assert (out[2] == 0.0).all()

    def test_per_channel_scales_differ(self):
        # Channels with different ranges should produce different scales.
        x = torch.zeros(4, 64)
        for c in range(4):
            x[c] = (c + 1) * 0.1 * torch.ones(64)
        out = ref(x)
        # Each channel's max abs value should differ.
        maxes = [out[c].abs().max().item() for c in range(4)]
        assert len(set(round(m, 4) for m in maxes)) == 4

    def test_single_channel_matches_pertensor(self):
        # When C_out=1, per-channel degenerates to per-tensor.
        from src.quantize import fake_quantize_int8_symmetric
        x = torch.randn(1, 16)
        per_ch = ref(x)
        per_ts = fake_quantize_int8_symmetric(x)
        assert torch.allclose(per_ch, per_ts, atol=1e-6)

    def test_quantization_error_bounded(self):
        torch.manual_seed(42)
        x = torch.randn(64, 256)
        out = ref(x)
        for c in range(64):
            scale_c = x[c].abs().max() / 127.0
            max_err = (out[c] - x[c]).abs().max().item()
            assert max_err <= 0.6 * scale_c.item() + 1e-6

    def test_dfl_shape(self):
        # DFL conv: [1, 16, 1, 1] — degenerate shape used in YOLOv8s
        x = torch.randn(1, 16, 1, 1)
        out = ref(x)
        assert out.shape == x.shape


# ── CUDA kernel matches reference ────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCudaMatchesReference:
    """GPU kernel output must match CPU reference within FP32 rounding tolerance."""

    def _check(self, shape, seed=0):
        torch.manual_seed(seed)
        x_cpu = torch.randn(*shape)
        x_gpu = x_cpu.cuda()
        expected = ref(x_cpu)
        got = fake_quant_perchannel(x_gpu).cpu()
        # Allow 1 ULP tolerance for rintf vs round differences.
        assert torch.allclose(expected, got, atol=1e-5), (
            f"shape={shape}: max diff = {(got - expected).abs().max():.2e}"
        )

    def test_large_conv_shape(self):
        self._check((256, 2304))  # [256, 256, 3, 3] flattened

    def test_medium_conv_shape(self):
        self._check((128, 1152))  # [128, 128, 3, 3]

    def test_small_conv_shape(self):
        self._check((64, 576))    # [64, 64, 3, 3]

    def test_1x1_conv_shape(self):
        self._check((256, 256))   # [256, 256, 1, 1]

    def test_dfl_shape(self):
        self._check((1, 16))      # DFL conv [1, 16, 1, 1]

    def test_4d_input(self):
        torch.manual_seed(7)
        x = torch.randn(64, 32, 3, 3).cuda()
        expected = ref(x.cpu())
        got = fake_quant_perchannel(x).cpu()
        assert torch.allclose(expected, got, atol=1e-5)

    def test_single_element_per_channel(self):
        # Spatial size = 1: each channel is a single scalar.
        self._check((32, 1))

    def test_spatial_size_not_multiple_of_block(self):
        # spatial_size = 100 is not a multiple of 32 — tests boundary handling.
        self._check((16, 100))

    def test_zero_channel_on_gpu(self):
        x = torch.randn(8, 64).cuda()
        x[3, :] = 0.0
        out = fake_quant_perchannel(x)
        assert (out[3].cpu() == 0.0).all()

    def test_all_channels_zero(self):
        x = torch.zeros(4, 64).cuda()
        out = fake_quant_perchannel(x)
        assert (out.cpu() == 0.0).all()


# ── apply_weight_fake_quant_perchannel ───────────────────────────────────────

class TestApplyAndRestore:
    def _model(self):
        return nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.Conv2d(8, 16, 1),
        )

    def test_apply_modifies_weights(self):
        model = self._model()
        originals = {n: c.weight.data.clone() for n, c in iter_conv_modules(model)}
        apply_weight_fake_quant_perchannel(model)
        for name, conv in iter_conv_modules(model):
            assert not torch.equal(conv.weight.data, originals[name])

    def test_restore_recovers_original(self):
        torch.manual_seed(3)
        model = self._model()
        originals = {n: c.weight.data.clone() for n, c in iter_conv_modules(model)}
        saved = apply_weight_fake_quant_perchannel(model)
        restore_weights(model, saved)
        for name, conv in iter_conv_modules(model):
            assert torch.equal(conv.weight.data, originals[name])

    def test_skip_layers_respected(self):
        model = self._model()
        original_1 = next(
            c.weight.data.clone() for n, c in iter_conv_modules(model) if n == "1"
        )
        apply_weight_fake_quant_perchannel(model, skip_layers={"1"})
        second_conv = next(c for n, c in iter_conv_modules(model) if n == "1")
        assert torch.equal(second_conv.weight.data, original_1)

    def test_per_channel_reduces_max_error_vs_pertensor(self):
        # Per-channel should have lower or equal max quantization error than per-tensor
        # for a tensor with heterogeneous channel magnitudes.
        torch.manual_seed(9)
        weight = torch.zeros(8, 16)
        for c in range(8):
            weight[c] = (c + 1) * 0.5 * torch.randn(16)  # channels differ in scale

        from src.quantize import fake_quantize_int8_symmetric
        per_tensor = fake_quantize_int8_symmetric(weight)
        per_channel = ref(weight)

        err_pt = (per_tensor - weight).abs().max().item()
        err_pc = (per_channel - weight).abs().max().item()
        assert err_pc <= err_pt + 1e-6, (
            f"per-channel error {err_pc:.4f} should be <= per-tensor {err_pt:.4f}"
        )
