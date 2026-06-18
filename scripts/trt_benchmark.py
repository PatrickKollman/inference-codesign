"""Layer 3, Step 2: Benchmark TRT FP16 engine latency using CUDA event timing.

Loads the serialized TRT engine and runs inference directly via the TRT 11.x
Python API — no ultralytics wrapper in the timing path. Uses torch tensors for
buffer management (no pycuda dependency).

The timing methodology matches the FP32 harness exactly:
  - 50 warmup iterations (TRT kernel selection outside timing window)
  - 200 timed iterations with per-iteration CUDA event timing
  - Same synchronization discipline as src/harness.py

This gives apples-to-apples comparison: FP32 eager 2.453ms vs TRT FP16 X.Xms.

Note: TRT engine has 3 output tensors (detection head at 3 scales, split by TRT
optimizer). All are allocated as buffers; only timing is recorded, not outputs.

Usage:
    python scripts/trt_benchmark.py [--engine results/yolov8s_fp16.trt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ENV_JSON = ROOT / "results" / "env.json"
RESULTS_DIR = ROOT / "results"

INPUT_SHAPE = (1, 3, 640, 640)
N_WARMUP = 50
N_REPS = 200
FP32_EAGER_MEAN_MS = 2.453  # from results/fp32_latency.json


def load_engine(path: Path):
    """Load a TRT engine, handling both raw bytes and ultralytics' wrapped format.

    ultralytics saves engines with a header: [4-byte meta_len LE][meta JSON][engine bytes].
    Raw TRT files start with the TRT magic tag directly.
    """
    import json
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(path, "rb") as f:
        first4 = f.read(4)
        meta_len = int.from_bytes(first4, byteorder="little")
        if 0 < meta_len < 10 * 1024 * 1024:  # plausible metadata length
            try:
                metadata = json.loads(f.read(meta_len).decode("utf-8"))
                engine_bytes = f.read()
                print(f"[load_engine] ultralytics format, metadata keys: {list(metadata.keys())}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                f.seek(0)
                engine_bytes = f.read()
                print("[load_engine] raw TRT format")
        else:
            f.seek(0)
            engine_bytes = f.read()
            print("[load_engine] raw TRT format")

    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {path}")
    return engine


def build_buffers(engine):
    """Allocate torch tensors for all engine I/O tensors."""
    import tensorrt as trt
    context = engine.create_execution_context()
    buffers = {}

    dtype_map = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF:  torch.float16,
        trt.DataType.INT8:  torch.int8,
        trt.DataType.INT32: torch.int32,
    }

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = dtype_map.get(engine.get_tensor_dtype(name), torch.float32)
        mode = "IN" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUT"
        tensor = torch.zeros(shape, dtype=dtype, device="cuda")
        context.set_tensor_address(name, tensor.data_ptr())
        buffers[name] = tensor
        print(f"  [{mode}] {name}: {list(shape)}  dtype={dtype}")

    return context, buffers


def benchmark_trt(engine) -> dict:
    context, buffers = build_buffers(engine)

    stream = torch.cuda.current_stream()
    stream_handle = stream.cuda_stream

    # Warmup — TRT lazy-compiles kernels on first few iterations
    print(f"\nWarmup ({N_WARMUP} iters)...")
    for _ in range(N_WARMUP):
        context.execute_async_v3(stream_handle=stream_handle)
    torch.cuda.synchronize()

    # CUDA event timing — identical methodology to FP32 harness
    print(f"Timing ({N_REPS} reps)...")
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    samples = []

    for _ in range(N_REPS):
        start_evt.record()
        context.execute_async_v3(stream_handle=stream_handle)
        end_evt.record()
        torch.cuda.synchronize()
        samples.append(start_evt.elapsed_time(end_evt))

    import numpy as np
    arr = np.array(samples)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "p99_ms":  float(np.percentile(arr, 99)),
        "std_ms":  float(np.std(arr)),
        "min_ms":  float(np.min(arr)),
        "max_ms":  float(np.max(arr)),
        "n_reps":  N_REPS,
        "n_warmup": N_WARMUP,
        "is_reportable": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        default=str(ROOT / "results" / "yolov8s_fp16.trt"),
        help="Path to serialized TRT engine.",
    )
    args = parser.parse_args()
    engine_path = Path(args.engine)

    if not engine_path.exists():
        print(f"Engine not found: {engine_path}")
        sys.exit(1)

    print(f"Loading: {engine_path}  ({engine_path.stat().st_size // 1024} KB)")
    engine = load_engine(engine_path)

    import tensorrt as trt
    n_in  = sum(1 for i in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT)
    n_out = engine.num_io_tensors - n_in
    print(f"Engine I/O: {n_in} input(s), {n_out} output(s)")
    print("Tensors:")

    timing = benchmark_trt(engine)

    precision = "int8" if "int8" in engine_path.name else "fp16"
    speedup = FP32_EAGER_MEAN_MS / timing["mean_ms"]
    print(f"\nTRT {precision.upper()} latency:")
    print(f"  mean  : {timing['mean_ms']:.3f} ms")
    print(f"  p50   : {timing['p50_ms']:.3f} ms")
    print(f"  p95   : {timing['p95_ms']:.3f} ms")
    print(f"  p99   : {timing['p99_ms']:.3f} ms")
    print(f"  std   : {timing['std_ms']:.3f} ms")
    print(f"  min   : {timing['min_ms']:.3f} ms   max: {timing['max_ms']:.3f} ms")
    print(f"\nSpeedup vs FP32 eager: {speedup:.2f}×")

    env_timestamp = None
    if ENV_JSON.exists():
        with open(ENV_JSON) as f:
            env_timestamp = json.load(f).get("timestamp_utc")

    precision_note = (
        "int8 (79 quantized nodes via ModelOpt; activations + weights calibrated on 512 COCO images)"
        if precision == "int8"
        else "fp16 (99.1% of nodes; 2 detection-head nodes kept FP32 by TRT)"
    )
    artifact = {
        "env_timestamp_utc": env_timestamp,
        "model": "yolov8s",
        "backend": f"tensorrt_{precision}",
        "engine": engine_path.name,
        "precision": precision_note,
        "input_shape": list(INPUT_SHAPE),
        "fp32_eager_mean_ms": FP32_EAGER_MEAN_MS,
        "speedup_vs_fp32_eager": round(speedup, 3),
        "timing": {k: v for k, v in timing.items()},
    }

    out = RESULTS_DIR / f"trt_{precision}_latency.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact: {out}")


if __name__ == "__main__":
    main()
