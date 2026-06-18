"""Day 4: Nsight Systems timeline capture for YOLOv8s FP32 forward pass.

Warmup runs BEFORE the cudaProfilerApi capture range so the nsys timeline
contains only steady-state inference — no JIT, no cuDNN heuristic selection.
Run via:
    nsys profile --capture-range=cudaProfilerApi --trace=cuda --stats=true \
        --output=results/day4_timeline --force-overwrite=true \
        python scripts/day4_profile_nsys.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import load_yolov8s

N_WARMUP = 50
N_CAPTURE = 20
INPUT_SHAPE = (1, 3, 640, 640)


def main() -> None:
    model = load_yolov8s(device="cuda")
    x = torch.zeros(INPUT_SHAPE, device="cuda", dtype=torch.float32)

    print(f"Warming up ({N_WARMUP} iters, outside capture range)...")
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(x)
    torch.cuda.synchronize()

    print(f"Starting nsys capture ({N_CAPTURE} iters)...")
    torch.cuda.cudart().cudaProfilerStart()

    with torch.no_grad():
        for _ in range(N_CAPTURE):
            model(x)
            torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStop()
    print("Capture complete.")


if __name__ == "__main__":
    main()
