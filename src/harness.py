"""CUDA timing harness for inference benchmarking.

Design principles
-----------------
* CUDA-event timing is the sole reportable path. Wall-clock (CPU/MPS) is
  structural testing only and is explicitly marked non-reportable.
* Per-iteration device synchronization measures latency, not throughput.
  This matches the single-camera, one-frame-at-a-time deployment model.
* Warmup (default 50 iters) covers: CUDA kernel JIT, cuDNN algorithm
  selection, GPU frequency ramp-up, and L2 cache warming. Timing starts
  only after a final synchronize() confirms all warmup work is complete.
* cudnn.benchmark=True is the default (production-realistic). The auto-tuner
  runs during warmup iterations and its choice persists into measurement.
  This setting is recorded in every artifact for reproducibility.
* Memory is measured after warmup so cuDNN workspace allocations are already
  in place and counted in the "model footprint" baseline, not as activations.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    n_warmup: int = 50
    n_reps: int = 200
    autocast_enabled: bool = False  # FP32 baseline; set True for AMP sweep
    cudnn_benchmark: bool = True    # production-realistic algorithm selection


@dataclass
class TimingResult:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n_reps: int
    n_warmup: int
    is_reportable: bool   # False on CPU/MPS — never publish these numbers
    samples_ms: list      # raw per-iteration timings for custom analysis


# ---------------------------------------------------------------------------
# Core statistics — kept separate so tests can verify it in isolation
# ---------------------------------------------------------------------------

def _compute_stats(
    samples: list,
    n_warmup: int,
    is_reportable: bool,
) -> TimingResult:
    arr = np.array(samples, dtype=np.float64)
    return TimingResult(
        mean_ms=float(np.mean(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        std_ms=float(np.std(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        n_reps=len(samples),
        n_warmup=n_warmup,
        is_reportable=is_reportable,
        samples_ms=list(samples),
    )


# ---------------------------------------------------------------------------
# Timing backends
# ---------------------------------------------------------------------------

def _time_cuda(
    fn: Callable[[], None],
    config: BenchmarkConfig,
    device: torch.device,
) -> list:
    """CUDA-event per-iteration latency timing. Authoritative measurement path.

    Events are placed on the default stream bracketing each fn() call.
    elapsed_time() gives GPU-side wall-clock between the two markers —
    independent of CPU dispatch overhead that runs concurrently.
    """
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

    try:
        for _ in range(config.n_warmup):
            fn()
        torch.cuda.synchronize(device)  # flush all warmup work before timing

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        samples = []

        for _ in range(config.n_reps):
            start_evt.record()
            fn()
            end_evt.record()
            # Synchronize after each iteration: measures latency (one frame at a
            # time), not throughput. Matches the single-camera deployment model.
            torch.cuda.synchronize(device)
            samples.append(start_evt.elapsed_time(end_evt))  # milliseconds

        return samples

    finally:
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark


def _time_cpu(
    fn: Callable[[], None],
    config: BenchmarkConfig,
) -> list:
    """Wall-clock timing via perf_counter. NOT a reported number.

    Used only to verify harness logic (call counts, artifact structure)
    on CPU/MPS where CUDA events are unavailable.
    """
    for _ in range(config.n_warmup):
        fn()

    samples = []
    for _ in range(config.n_reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)  # ms

    return samples


# ---------------------------------------------------------------------------
# Public benchmark entry point
# ---------------------------------------------------------------------------

def benchmark(
    fn: Callable[[], None],
    config: BenchmarkConfig,
    device: str | torch.device = "cuda",
) -> TimingResult:
    """Time a zero-argument callable under no_grad and optional autocast.

    The caller closes over the model and input tensor::

        x = torch.zeros((1, 3, 640, 640), device="cuda")
        result = benchmark(lambda: model(x), config)

    The fn must not include data loading or preprocessing — benchmark the
    neural network forward pass only.
    """
    _device = torch.device(device)
    is_cuda = _device.type == "cuda"

    def _wrapped() -> None:
        with torch.no_grad():
            if is_cuda:
                with torch.amp.autocast(device_type="cuda", enabled=config.autocast_enabled):
                    fn()
            else:
                fn()

    if is_cuda:
        samples = _time_cuda(_wrapped, config, _device)
    else:
        samples = _time_cpu(_wrapped, config)

    return _compute_stats(samples, config.n_warmup, is_reportable=is_cuda)


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def measure_memory(
    fn: Callable[[], None],
    device: str | torch.device = "cuda",
) -> dict:
    """Measure GPU memory footprint during a forward pass.

    Call this AFTER benchmark() so cuDNN workspace allocations are already
    in place. Those allocations are then counted in `allocated_before_mb`
    (model + workspace baseline), not as activation memory.

    Returns a dict with keys:
        allocated_before_mb: model weights + persistent workspace
        peak_during_mb:      peak during the measured forward pass
        activation_mb:       peak_during_mb - allocated_before_mb
        is_cuda:             False when called on CPU (all values zero)
    """
    _device = torch.device(device)
    if _device.type != "cuda":
        return {
            "allocated_before_mb": 0.0,
            "peak_during_mb": 0.0,
            "activation_mb": 0.0,
            "is_cuda": False,
        }

    torch.cuda.synchronize(_device)
    before_bytes = torch.cuda.memory_allocated(_device)
    torch.cuda.reset_peak_memory_stats(_device)

    with torch.no_grad():
        fn()

    torch.cuda.synchronize(_device)
    peak_bytes = torch.cuda.max_memory_allocated(_device)

    to_mb = 1.0 / (1024 ** 2)
    return {
        "allocated_before_mb": round(before_bytes * to_mb, 2),
        "peak_during_mb": round(peak_bytes * to_mb, 2),
        "activation_mb": round((peak_bytes - before_bytes) * to_mb, 2),
        "is_cuda": True,
    }


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_timing_artifact(
    result: TimingResult,
    artifact_name: str,
    metadata: dict | None = None,
    env_json_path: Path | None = None,
) -> Path:
    """Write a timing artifact to results/{artifact_name}.json.

    Every artifact links back to the env provenance timestamp from
    env.json so numbers can always be traced to a verified environment.

    Args:
        result:        TimingResult from benchmark().
        artifact_name: Filename stem, e.g. "fp32_latency".
        metadata:      Any additional fields (model, input shape, etc.).
        env_json_path: Path to env.json for provenance linking.

    Returns:
        Path to the written artifact.
    """
    env_timestamp = None
    if env_json_path is not None and env_json_path.exists():
        with open(env_json_path) as f:
            env_timestamp = json.load(f).get("timestamp_utc")
    elif env_json_path is not None:
        # Caller explicitly provided a path that doesn't exist — flag it
        env_timestamp = f"[MISSING: {env_json_path}]"

    artifact = {
        "env_timestamp_utc": env_timestamp,
        **(metadata or {}),
        "timing": {
            "mean_ms": result.mean_ms,
            "p50_ms": result.p50_ms,
            "p95_ms": result.p95_ms,
            "p99_ms": result.p99_ms,
            "std_ms": result.std_ms,
            "min_ms": result.min_ms,
            "max_ms": result.max_ms,
            "n_reps": result.n_reps,
            "n_warmup": result.n_warmup,
            "is_reportable": result.is_reportable,
        },
        "samples_ms": result.samples_ms,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{artifact_name}.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    return out_path
