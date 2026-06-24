"""Export YOLOv8s to TensorRT INT8 engine.

Uses ultralytics' built-in TRT export, which handles:
  - ONNX graph export and simplification
  - INT8 calibration using the provided dataset (activation scale collection)
  - Engine serialization

INT8 calibration in TRT
-----------------------
TRT INT8 requires per-tensor activation scales (not just weight scales like
our fake-quant sweep). The calibrator feeds real image batches through the
network, records min/max activations at each tensor, and derives scales.
This is why the fake-quant accuracy numbers differ slightly from TRT INT8 —
the activation quantization noise is additive.

Engine output: results/yolov8s_int8.trt

Usage:
    python scripts/trt_int8_build.py [--data-dir data/coco]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ENGINE_PATH = ROOT / "results" / "yolov8s_int8.trt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "coco"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if ENGINE_PATH.exists():
        print(f"Engine already exists: {ENGINE_PATH}  ({ENGINE_PATH.stat().st_size // 1024} KB)")
        print("Delete it to rebuild.")
        return

    # Write a concrete coco yaml for ultralytics (needs a real file, not a tmp)
    from src.eval import _make_coco_yaml
    yaml_content = _make_coco_yaml(data_dir)
    tmp_fd, tmp_yaml = tempfile.mkstemp(suffix=".yaml", prefix="coco_trt_")
    with os.fdopen(tmp_fd, "w") as f:
        f.write(yaml_content)

    try:
        from ultralytics import YOLO
        model = YOLO("yolov8s.pt")

        print("Building TRT INT8 engine (5–15 min: ONNX export + calibration + build)...")
        # ultralytics export to TRT with INT8 calibration
        exported = model.export(
            format="engine",
            imgsz=640,
            int8=True,
            data=tmp_yaml,
            device=0,
            workspace=4,   # GB of TRT workspace
            simplify=True,
        )
    finally:
        os.unlink(tmp_yaml)

    exported_path = Path(exported) if exported else Path("yolov8s.engine")

    # Move to results/ with our naming convention
    ENGINE_PATH.parent.mkdir(exist_ok=True)
    if exported_path.exists() and exported_path != ENGINE_PATH:
        exported_path.rename(ENGINE_PATH)

    print(f"Engine: {ENGINE_PATH}  ({ENGINE_PATH.stat().st_size // (1024**2):.1f} MB)")
    print("Next: python scripts/trt_benchmark.py")


if __name__ == "__main__":
    main()
