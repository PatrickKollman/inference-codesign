"""Unit tests for src/harness.py.

All tests run on CPU — no CUDA required. They verify harness logic
(call counts, stat computation, artifact structure, provenance linking)
without producing any reportable numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import src.harness as harness_module
from src.harness import (
    BenchmarkConfig,
    TimingResult,
    _compute_stats,
    benchmark,
    measure_memory,
    save_timing_artifact,
)


# ---------------------------------------------------------------------------
# _compute_stats — isolated from any device
# ---------------------------------------------------------------------------

class TestComputeStats:
    def test_mean(self):
        result = _compute_stats([1.0, 2.0, 3.0, 4.0, 5.0], n_warmup=0, is_reportable=True)
        assert result.mean_ms == pytest.approx(3.0)

    def test_min_max(self):
        result = _compute_stats([1.0, 2.0, 5.0], n_warmup=0, is_reportable=False)
        assert result.min_ms == pytest.approx(1.0)
        assert result.max_ms == pytest.approx(5.0)

    def test_std_uniform(self):
        result = _compute_stats([10.0] * 100, n_warmup=50, is_reportable=True)
        assert result.std_ms == pytest.approx(0.0, abs=1e-9)

    def test_percentiles_uniform(self):
        result = _compute_stats([10.0] * 100, n_warmup=50, is_reportable=True)
        assert result.p50_ms == pytest.approx(10.0)
        assert result.p95_ms == pytest.approx(10.0)
        assert result.p99_ms == pytest.approx(10.0)

    def test_percentiles_known_distribution(self):
        # 100 values: 0, 1, 2, ..., 99
        samples = list(range(100))
        result = _compute_stats(samples, n_warmup=0, is_reportable=False)
        assert result.p50_ms == pytest.approx(np.percentile(samples, 50))
        assert result.p95_ms == pytest.approx(np.percentile(samples, 95))
        assert result.p99_ms == pytest.approx(np.percentile(samples, 99))

    def test_samples_preserved(self):
        samples = [1.0, 2.0, 3.0]
        result = _compute_stats(samples, n_warmup=5, is_reportable=False)
        assert result.samples_ms == samples
        assert result.n_reps == 3
        assert result.n_warmup == 5

    def test_is_reportable_propagated(self):
        r_true = _compute_stats([1.0], n_warmup=0, is_reportable=True)
        r_false = _compute_stats([1.0], n_warmup=0, is_reportable=False)
        assert r_true.is_reportable is True
        assert r_false.is_reportable is False


# ---------------------------------------------------------------------------
# benchmark() on CPU — verifies call counts and structural invariants
# ---------------------------------------------------------------------------

class TestBenchmarkCPU:
    def test_warmup_and_rep_call_count(self):
        calls = []
        def fn():
            calls.append(1)

        config = BenchmarkConfig(n_warmup=7, n_reps=13)
        benchmark(fn, config, device="cpu")

        # no_grad wrapper is transparent to call counting
        assert len(calls) == 20  # 7 warmup + 13 measured

    def test_rep_count_in_result(self):
        config = BenchmarkConfig(n_warmup=3, n_reps=17)
        result = benchmark(lambda: None, config, device="cpu")
        assert result.n_reps == 17
        assert result.n_warmup == 3

    def test_samples_length_matches_n_reps(self):
        config = BenchmarkConfig(n_warmup=5, n_reps=30)
        result = benchmark(lambda: None, config, device="cpu")
        assert len(result.samples_ms) == 30

    def test_not_reportable_on_cpu(self):
        result = benchmark(lambda: None, BenchmarkConfig(n_warmup=1, n_reps=5), device="cpu")
        assert result.is_reportable is False

    def test_all_timing_fields_non_negative(self):
        result = benchmark(lambda: None, BenchmarkConfig(n_warmup=2, n_reps=10), device="cpu")
        for field in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "std_ms", "min_ms", "max_ms"):
            assert getattr(result, field) >= 0.0, f"{field} is negative"

    def test_p95_gte_p50_gte_min(self):
        result = benchmark(lambda: None, BenchmarkConfig(n_warmup=2, n_reps=50), device="cpu")
        assert result.p95_ms >= result.p50_ms
        assert result.p50_ms >= result.min_ms
        assert result.max_ms >= result.p99_ms

    def test_fn_exception_propagates(self):
        def bad_fn():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            benchmark(bad_fn, BenchmarkConfig(n_warmup=0, n_reps=1), device="cpu")


# ---------------------------------------------------------------------------
# measure_memory — CPU path returns zero/non-cuda sentinel
# ---------------------------------------------------------------------------

class TestMeasureMemoryCPU:
    def test_returns_zeros_on_cpu(self):
        mem = measure_memory(lambda: None, device="cpu")
        assert mem["is_cuda"] is False
        assert mem["allocated_before_mb"] == 0.0
        assert mem["peak_during_mb"] == 0.0
        assert mem["activation_mb"] == 0.0

    def test_required_keys_present(self):
        mem = measure_memory(lambda: None, device="cpu")
        for key in ("allocated_before_mb", "peak_during_mb", "activation_mb", "is_cuda"):
            assert key in mem, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# save_timing_artifact — file structure and provenance
# ---------------------------------------------------------------------------

class TestSaveTimingArtifact:
    def _make_result(self) -> TimingResult:
        return _compute_stats([1.0, 2.0, 3.0], n_warmup=1, is_reportable=False)

    def test_creates_json_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(harness_module, "RESULTS_DIR", tmp_path)
        result = self._make_result()
        out = save_timing_artifact(result, "test_artifact")
        assert out.exists()
        assert out.suffix == ".json"

    def test_timing_block_has_all_required_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr(harness_module, "RESULTS_DIR", tmp_path)
        out = save_timing_artifact(self._make_result(), "test_keys")
        with open(out) as f:
            artifact = json.load(f)

        assert "timing" in artifact
        required_timing_keys = {
            "mean_ms", "p50_ms", "p95_ms", "p99_ms",
            "std_ms", "min_ms", "max_ms",
            "n_reps", "n_warmup", "is_reportable",
        }
        missing = required_timing_keys - set(artifact["timing"])
        assert not missing, f"Missing timing keys: {missing}"

    def test_samples_included_in_artifact(self, tmp_path, monkeypatch):
        monkeypatch.setattr(harness_module, "RESULTS_DIR", tmp_path)
        out = save_timing_artifact(self._make_result(), "test_samples")
        with open(out) as f:
            artifact = json.load(f)
        assert "samples_ms" in artifact
        assert len(artifact["samples_ms"]) == 3

    def test_metadata_merged_into_artifact(self, tmp_path, monkeypatch):
        monkeypatch.setattr(harness_module, "RESULTS_DIR", tmp_path)
        meta = {"model": "yolov8s", "precision": "fp32", "input_shape": [1, 3, 640, 640]}
        out = save_timing_artifact(self._make_result(), "test_meta", metadata=meta)
        with open(out) as f:
            artifact = json.load(f)
        assert artifact["model"] == "yolov8s"
        assert artifact["precision"] == "fp32"
        assert artifact["input_shape"] == [1, 3, 640, 640]

    def test_env_provenance_linked_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(harness_module, "RESULTS_DIR", tmp_path)
        env_json = tmp_path / "env.json"
        env_json.write_text(json.dumps({"timestamp_utc": "2026-01-01T00:00:00+00:00"}))

        out = save_timing_artifact(self._make_result(), "test_prov", env_json_path=env_json)
        with open(out) as f:
            artifact = json.load(f)
        assert artifact["env_timestamp_utc"] == "2026-01-01T00:00:00+00:00"

    def test_env_provenance_null_when_not_provided(self, tmp_path, monkeypatch):
        monkeypatch.setattr(harness_module, "RESULTS_DIR", tmp_path)
        out = save_timing_artifact(self._make_result(), "test_no_prov")
        with open(out) as f:
            artifact = json.load(f)
        assert artifact["env_timestamp_utc"] is None

    def test_missing_env_file_flagged_not_silenced(self, tmp_path, monkeypatch):
        monkeypatch.setattr(harness_module, "RESULTS_DIR", tmp_path)
        missing = tmp_path / "nonexistent_env.json"
        out = save_timing_artifact(self._make_result(), "test_missing", env_json_path=missing)
        with open(out) as f:
            artifact = json.load(f)
        # Should contain the path string so the caller knows it's missing, not just null
        assert artifact["env_timestamp_utc"] is not None
        assert "MISSING" in str(artifact["env_timestamp_utc"])

    def test_is_reportable_false_preserved_in_artifact(self, tmp_path, monkeypatch):
        monkeypatch.setattr(harness_module, "RESULTS_DIR", tmp_path)
        out = save_timing_artifact(self._make_result(), "test_reportable")
        with open(out) as f:
            artifact = json.load(f)
        assert artifact["timing"]["is_reportable"] is False
