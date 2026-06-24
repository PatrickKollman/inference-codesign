"""Benchmark per-channel INT8 fake-quantize kernel vs PyTorch baseline.

Measures throughput (GB/s) and positions results on the roofline.
Tested shapes match profiled YOLOv8s conv layers.

Usage:
    python scripts/kernel_benchmark.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ENV_JSON    = ROOT / "results" / "env.json"
RESULTS_DIR = ROOT / "results"

# RTX 4090 specs (Ada Lovelace)
PEAK_BW_GBs   = 1008.0   # GB/s DRAM bandwidth
FP32_SIZE     = 4         # bytes per element

N_WARMUP = 50
N_REPS   = 1000

# Representative YOLOv8s conv shapes [C_out, C_in, kH, kW].
# From profiling (docs/profiling_analysis.md): these are the expensive layers.
SHAPES = [
    ((256, 256, 3, 3), "backbone_3x3_256"),
    ((128, 128, 3, 3), "neck_3x3_128"),
    ((64,  64,  3, 3), "small_3x3_64"),
    ((256, 256, 1, 1), "backbone_1x1_256"),
    ((128,  64, 1, 1), "neck_1x1_128x64"),
    ((1,   16,  1, 1), "dfl_conv"),       # DFL edge case: C_out=1
]


def pytorch_baseline(weight: torch.Tensor) -> torch.Tensor:
    """Per-channel INT8 fake-quant via PyTorch ops (no custom kernel)."""
    original_shape = weight.shape
    C_out = original_shape[0]
    w2d = weight.reshape(C_out, -1)
    scales = w2d.abs().max(dim=1).values / 127.0
    scales = scales.clamp(min=1e-8)
    w_q = (w2d / scales.unsqueeze(1)).round_().clamp_(-128, 127).mul_(scales.unsqueeze(1))
    return w_q.reshape(original_shape)


def time_fn(fn, weight, n_warmup, n_reps):
    """CUDA event timing matching the harness methodology."""
    for _ in range(n_warmup):
        fn(weight)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(n_reps):
        start.record()
        fn(weight)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return np.array(samples)


def throughput_gbs(shape, mean_ms):
    """Effective memory bandwidth: 2 reads + 1 write = 3 passes over the tensor."""
    n_elements = 1
    for d in shape:
        n_elements *= d
    bytes_rw = 3 * n_elements * FP32_SIZE   # read twice (reduction + quant), write once
    gb = bytes_rw / 1e9
    sec = mean_ms / 1e3
    return gb / sec


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — benchmark must run on the pod.")
        sys.exit(1)

    from src.cuda.fake_quant_perchannel import fake_quant_perchannel

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Peak DRAM BW: {PEAK_BW_GBs} GB/s")
    print(f"Warmup: {N_WARMUP}  Reps: {N_REPS}\n")

    # Force JIT compile before timing loop.
    _dummy = torch.randn(4, 16, device="cuda")
    fake_quant_perchannel(_dummy)
    torch.cuda.synchronize()
    print("Kernel compiled.\n")

    results = []

    header = f"{'Shape':<28} {'Method':<12} {'mean µs':>9} {'p99 µs':>9} {'GB/s':>8} {'%peak':>7}"
    print(header)
    print("-" * len(header))

    for shape, label in SHAPES:
        weight = torch.randn(*shape, device="cuda")

        for method_name, fn in [
            ("PyTorch",     pytorch_baseline),
            ("CUDA kernel", fake_quant_perchannel),
        ]:
            samples_ms = time_fn(fn, weight, N_WARMUP, N_REPS)
            mean_us    = float(np.mean(samples_ms)) * 1e3
            p99_us     = float(np.percentile(samples_ms, 99)) * 1e3
            bw         = throughput_gbs(shape, float(np.mean(samples_ms)))
            pct_peak   = bw / PEAK_BW_GBs * 100

            shape_str = f"{list(shape)}"
            print(f"{shape_str:<28} {method_name:<12} {mean_us:>9.2f} {p99_us:>9.2f} {bw:>8.1f} {pct_peak:>6.1f}%")

            results.append({
                "shape": list(shape),
                "label": label,
                "method": method_name,
                "mean_us": round(mean_us, 3),
                "p99_us":  round(p99_us, 3),
                "throughput_gbs": round(bw, 2),
                "pct_peak_bw": round(pct_peak, 2),
            })

        # Compute speedup for this shape.
        pt_mean  = next(r["mean_us"] for r in results if r["label"] == label and r["method"] == "PyTorch")
        cu_mean  = next(r["mean_us"] for r in results if r["label"] == label and r["method"] == "CUDA kernel")
        speedup = pt_mean / cu_mean
        print(f"  → speedup: {speedup:.2f}×")
        results[-1]["speedup_vs_pytorch"] = round(speedup, 3)
        results[-2]["speedup_vs_pytorch"] = None
        print()

    # ── Total time for all YOLOv8s conv layers (QAT forward pass cost) ──────
    print("\n── Total fake-quant overhead across all 64 YOLOv8s conv layers ──")
    from ultralytics import YOLO
    from src.quantize import iter_conv_modules

    model = YOLO("yolov8s.pt").model.eval().cuda()

    def quant_all_pytorch(m):
        for _, conv in iter_conv_modules(m):
            pytorch_baseline(conv.weight.data)

    def quant_all_cuda(m):
        for _, conv in iter_conv_modules(m):
            fake_quant_perchannel(conv.weight.data)

    for method_name, fn in [
        ("PyTorch",     quant_all_pytorch),
        ("CUDA kernel", quant_all_cuda),
    ]:
        for _ in range(10):
            fn(model)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            fn(model)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mean_ms = (t1 - t0) / 100 * 1e3
        print(f"  {method_name:<12}: {mean_ms:.3f} ms / forward pass")

    # ── Save artifact ────────────────────────────────────────────────────────
    env_timestamp = None
    if ENV_JSON.exists():
        with open(ENV_JSON) as f:
            env_timestamp = json.load(f).get("timestamp_utc")

    artifact = {
        "env_timestamp_utc": env_timestamp,
        "model": "yolov8s",
        "kernel": "fake_quant_perchannel",
        "description": "Per-channel INT8 fake-quantize: CUDA kernel vs PyTorch baseline",
        "peak_bw_gbs": PEAK_BW_GBs,
        "n_warmup": N_WARMUP,
        "n_reps": N_REPS,
        "per_shape_results": results,
    }

    out = RESULTS_DIR / "kernel_benchmark.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact: {out}")


if __name__ == "__main__":
    main()
