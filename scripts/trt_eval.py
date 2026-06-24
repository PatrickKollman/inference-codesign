"""Evaluate TRT engine mAP on COCO val2017.

Loads the TRT engine via ultralytics' YOLO wrapper and runs the standard
val pipeline. This gives the measured mAP for the TRT INT8 Pareto point,
replacing the fake-quant approximation.

Usage:
    python scripts/trt_eval.py [--data-dir data/coco]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ENV_JSON = ROOT / "results" / "env.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "coco"))
    parser.add_argument("--engine", default=str(ROOT / "results" / "yolov8s_fp16.trt"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    engine_path = Path(args.engine)

    if not engine_path.exists():
        print(f"Engine not found: {engine_path}")
        sys.exit(1)

    from ultralytics import YOLO
    from src.eval import _make_coco_yaml, parse_metrics, save_eval_artifact

    # ultralytics requires .engine extension to recognize TRT format
    if engine_path.suffix != ".engine":
        import shutil
        engine_alias = engine_path.with_suffix(".engine")
        if not engine_alias.exists():
            shutil.copy(engine_path, engine_alias)
        engine_path = engine_alias

    model = YOLO(str(engine_path), task="detect")

    yaml_content = _make_coco_yaml(data_dir)
    tmp_fd, tmp_yaml = tempfile.mkstemp(suffix=".yaml", prefix="coco_trt_eval_")
    with os.fdopen(tmp_fd, "w") as f:
        f.write(yaml_content)

    try:
        print(f"Running COCO val2017 with TRT engine: {engine_path.name}...")
        result = model.val(
            data=tmp_yaml,
            imgsz=640,
            device=0,
            batch=16,
            workers=4,
            verbose=False,
            save=False,
        )
    finally:
        os.unlink(tmp_yaml)

    metrics = parse_metrics(result)
    print("\nTRT INT8 mAP results:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}" if v is not None else f"  {k:15s}: N/A")

    fp32_map = 0.4442
    print(f"\nmAP50-95 drop vs FP32: {metrics['mAP50_95'] - fp32_map:+.4f}")

    precision = "fp16" if "fp16" in engine_path.name else "int8"
    out = save_eval_artifact(
        metrics,
        artifact_name=f"trt_{precision}_accuracy",
        metadata={
            "model": "yolov8s",
            "backend": f"tensorrt_{precision}",
            "engine": engine_path.name,
            "dataset": "coco_val2017",
            "n_images": 5000,
            "fp32_baseline_mAP50_95": fp32_map,
        },
        env_json_path=ENV_JSON,
    )
    print(f"\nArtifact: {out}")


if __name__ == "__main__":
    main()
