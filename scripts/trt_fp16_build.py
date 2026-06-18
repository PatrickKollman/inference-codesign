"""Layer 3, Step 0: Export YOLOv8s to TensorRT FP16 engine.

FP16 export does not require calibration data — TRT converts weights and
activations to FP16 analytically (no dataset pass needed). Build time is
~5–10 minutes: ONNX export + TRT engine build with fp16 mode enabled.

Engine output: results/yolov8s_fp16.trt

Usage:
    python scripts/trt_fp16_build.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ENGINE_PATH = ROOT / "results" / "yolov8s_fp16.trt"


def main() -> None:
    if ENGINE_PATH.exists():
        print(f"Engine already exists: {ENGINE_PATH}  ({ENGINE_PATH.stat().st_size // (1024**2):.1f} MB)")
        print("Delete it to rebuild.")
        return

    from ultralytics import YOLO
    model = YOLO("yolov8s.pt")

    print("Building TRT FP16 engine (~5–10 min: ONNX export + TRT build)...")
    exported = model.export(
        format="engine",
        imgsz=640,
        half=True,
        device=0,
        workspace=4,
        simplify=True,
    )

    exported_path = Path(exported) if exported else Path("yolov8s.engine")

    ENGINE_PATH.parent.mkdir(exist_ok=True)
    if exported_path.exists() and exported_path != ENGINE_PATH:
        shutil.move(str(exported_path), ENGINE_PATH)

    print(f"Engine: {ENGINE_PATH}  ({ENGINE_PATH.stat().st_size // (1024**2):.1f} MB)")
    print("Next: python scripts/trt_benchmark.py --engine results/yolov8s_fp16.trt")


if __name__ == "__main__":
    main()
