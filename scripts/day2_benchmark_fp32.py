"""Day 2: FP32 baseline latency + memory benchmark.

Produces results/day2_baseline_fp32.json — the measurement anchor for all
downstream comparisons (INT8, AMP, TensorRT). Every later optimization result
reports its improvement relative to this file's timing block.

Run on RunPod after Day 1 artifacts are committed:
    python scripts/day2_benchmark_fp32.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.harness import BenchmarkConfig, benchmark, measure_memory, save_timing_artifact
from src.model import load_yolov8s, parameter_count

ENV_JSON = ROOT / "results" / "day1_env.json"
INPUT_SHAPE = (1, 3, 640, 640)
DEVICE = "cuda"

CONFIG = BenchmarkConfig(
    n_warmup=50,
    n_reps=200,
    autocast_enabled=False,   # FP32 baseline — no autocast
    cudnn_benchmark=True,     # production-realistic; auto-tuner runs during warmup
)


def main() -> None:
    device = DEVICE if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: No CUDA device found. Results will NOT be reportable.")

    print(f"Device: {device}")
    model = load_yolov8s(device=device)
    n_params = parameter_count(model)
    print(f"Parameters: {n_params:,}")

    x = torch.zeros(INPUT_SHAPE, device=device, dtype=torch.float32)
    fn = lambda: model(x)  # noqa: E731

    # --- Timing ---
    print(f"\nBenchmarking ({CONFIG.n_warmup} warmup, {CONFIG.n_reps} reps, FP32)...")
    result = benchmark(fn, CONFIG, device=device)

    print(f"  mean  : {result.mean_ms:.3f} ms")
    print(f"  p50   : {result.p50_ms:.3f} ms")
    print(f"  p95   : {result.p95_ms:.3f} ms")
    print(f"  p99   : {result.p99_ms:.3f} ms")
    print(f"  std   : {result.std_ms:.3f} ms")
    print(f"  min   : {result.min_ms:.3f} ms    max: {result.max_ms:.3f} ms")
    print(f"  reportable: {result.is_reportable}")

    # --- Memory (measure after timing; cuDNN workspace already allocated) ---
    print("\nMeasuring memory...")
    mem = measure_memory(fn, device=device)
    print(f"  model + workspace : {mem['allocated_before_mb']:.1f} MB")
    print(f"  peak during fwd   : {mem['peak_during_mb']:.1f} MB")
    print(f"  activation memory : {mem['activation_mb']:.1f} MB")

    # --- Artifact ---
    metadata = {
        "model": "yolov8s",
        "precision": "fp32",
        "input_shape": list(INPUT_SHAPE),
        "n_parameters": n_params,
        "device": device,
        "cudnn_benchmark": CONFIG.cudnn_benchmark,
        "memory": mem,
    }
    out = save_timing_artifact(
        result,
        artifact_name="day2_baseline_fp32",
        metadata=metadata,
        env_json_path=ENV_JSON,
    )
    print(f"\nArtifact: {out}")
    if not result.is_reportable:
        print("WARNING: is_reportable=False — do not cite these numbers.")


if __name__ == "__main__":
    main()
