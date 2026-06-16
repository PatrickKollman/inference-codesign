"""Unit tests for src/quantize.py. No GPU or ultralytics required."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.quantize import (
    apply_weight_fake_quant,
    count_conv_params,
    fake_quantize_int8_symmetric,
    iter_conv_modules,
    restore_weights,
)


class TestFakeQuantizeInt8Symmetric:
    def test_output_dtype_preserved(self):
        x = torch.randn(8, 3, 3, 3, dtype=torch.float32)
        out = fake_quantize_int8_symmetric(x)
        assert out.dtype == torch.float32

    def test_values_within_int8_range(self):
        x = torch.randn(16, 16, 3, 3) * 0.5
        scale = x.abs().max() / 127.0
        out = fake_quantize_int8_symmetric(x)
        # All output values must be integer multiples of scale within [-128,127]*scale
        assert out.abs().max() <= 127.0 * scale + 1e-5

    def test_quantization_reduces_unique_values(self):
        x = torch.linspace(-1.0, 1.0, 1000)
        out = fake_quantize_int8_symmetric(x)
        # INT8 has at most 256 unique values; a range of 1000 floats → many collapse
        assert out.unique().numel() <= 256

    def test_zero_preserved(self):
        x = torch.randn(4, 4, 3, 3)
        # Zero maps to 0 in symmetric quantization
        x[0, 0, 0, 0] = 0.0
        out = fake_quantize_int8_symmetric(x)
        assert out[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)

    def test_empty_tensor_passthrough(self):
        x = torch.zeros(0)
        out = fake_quantize_int8_symmetric(x)
        assert out.shape == x.shape

    def test_large_tensor_close_to_original(self):
        # Quantization error should be small relative to signal range
        torch.manual_seed(0)
        x = torch.randn(64, 64, 3, 3)
        out = fake_quantize_int8_symmetric(x)
        # Max absolute error ≤ 0.5 * scale ≈ 0.5 * max(|x|) / 127
        scale = x.abs().max() / 127.0
        assert (out - x).abs().max() <= 0.6 * scale  # 0.6 for rounding safety

    def test_scale_based_on_max_abs(self):
        x = torch.tensor([0.0, 0.5, -1.27])
        # scale = 1.27 / 127 = 0.01, so -1.27 → -127 → -1.27 exactly
        out = fake_quantize_int8_symmetric(x)
        assert out[-1].item() == pytest.approx(-1.27, abs=1e-5)


class TestIterConvModules:
    def _toy_model(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 1),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
            ),
        )

    def test_finds_all_conv2d(self):
        model = self._toy_model()
        convs = list(iter_conv_modules(model))
        assert len(convs) == 3

    def test_yields_name_and_module_pairs(self):
        model = self._toy_model()
        for name, module in iter_conv_modules(model):
            assert isinstance(name, str)
            assert isinstance(module, nn.Conv2d)

    def test_names_are_fully_qualified(self):
        model = self._toy_model()
        names = [name for name, _ in iter_conv_modules(model)]
        # Nested conv should have a dotted name
        assert any("." in n for n in names)

    def test_empty_model_yields_nothing(self):
        model = nn.Sequential(nn.ReLU(), nn.BatchNorm2d(3))
        convs = list(iter_conv_modules(model))
        assert convs == []


class TestCountConvParams:
    def test_correct_param_count(self):
        conv = nn.Conv2d(3, 16, 3)
        # weight: 16 * 3 * 3 * 3 = 432
        assert count_conv_params(conv) == 432

    def test_depthwise_conv(self):
        conv = nn.Conv2d(32, 32, 3, groups=32)
        # weight: 32 * 1 * 3 * 3 = 288
        assert count_conv_params(conv) == 288


class TestApplyAndRestoreWeights:
    def _model(self):
        return nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.Conv2d(8, 16, 1),
        )

    def test_apply_modifies_weights(self):
        model = self._model()
        originals_copy = {n: c.weight.data.clone() for n, c in iter_conv_modules(model)}
        apply_weight_fake_quant(model)
        # Weights should be modified (quantized)
        for name, conv in iter_conv_modules(model):
            assert not torch.equal(conv.weight.data, originals_copy[name])

    def test_restore_recovers_original(self):
        torch.manual_seed(42)
        model = self._model()
        originals_copy = {n: c.weight.data.clone() for n, c in iter_conv_modules(model)}

        saved = apply_weight_fake_quant(model)
        restore_weights(model, saved)

        for name, conv in iter_conv_modules(model):
            assert torch.equal(conv.weight.data, originals_copy[name])

    def test_restored_weights_match_original_exactly(self):
        torch.manual_seed(7)
        model = nn.Sequential(nn.Conv2d(4, 8, 3))
        before = next(iter_conv_modules(model))[1].weight.data.clone()
        saved = apply_weight_fake_quant(model)
        restore_weights(model, saved)
        after = next(iter_conv_modules(model))[1].weight.data
        assert torch.equal(before, after)
