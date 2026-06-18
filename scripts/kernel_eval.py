"""Layer 2: Evaluate per-channel fake-quant mAP vs per-tensor on COCO val2017.

Compares three configurations:
  - Per-tensor all (64/64 conv, Day 5 baseline):   measured 0.4399
  - Per-channel smart (63/64, DFL excluded):        this script
  - FP32 reference:                                 measured 0.4442

Usage:
    python scripts/kernel_eval.py [--data-dir data/coco]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ENV_JSON = ROOT / "results" / "env.json"

FP32_MAP       = 0.4442   # fp32_accuracy.json
PERTENSOR_MAP  = 0.4399   # quant_pertensor_all.json  (all 64 conv)
SMART_PT_MAP   = 0.4425   # quant_pertensor_smart.json  (63/64, DFL excluded)

DFL_LAYER = "model.22.dfl.conv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "coco"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not torch.cuda.is_available():
        print("CUDA required — run on the pod.")
        sys.exit(1)

    from ultralytics import YOLO
    from src.cuda.fake_quant_perchannel import apply_weight_fake_quant_perchannel
    from src.eval import _make_coco_yaml, parse_metrics, save_eval_artifact
    from src.quantize import restore_weights

    print("Loading YOLOv8s...")
    model = YOLO("yolov8s.pt")
    # Keep weights on CPU for fake-quant; ultralytics val moves to device=0 itself.

    print(f"Applying per-channel fake-quant (all conv except {DFL_LAYER})...")
    saved = apply_weight_fake_quant_perchannel(
        model.model,
        skip_layers={DFL_LAYER},
    )

    quant_layers = len(saved)
    print(f"Quantized {quant_layers} conv layers (DFL excluded).")

    import os
    import tempfile
    from src.eval import _make_coco_yaml

    yaml_content = _make_coco_yaml(data_dir)
    tmp_fd, tmp_yaml = tempfile.mkstemp(suffix=".yaml", prefix="coco_perchannel_")
    with os.fdopen(tmp_fd, "w") as f:
        f.write(yaml_content)

    print("\nRunning COCO val2017 (5000 images)...")
    try:
        result = model.val(
            data=tmp_yaml,
            imgsz=640,
            device=0,
            batch=16,
            workers=4,
            verbose=True,
            save=False,
        )
    finally:
        os.unlink(tmp_yaml)

    metrics = parse_metrics(result)
    restore_weights(model.model, saved)
    print("Weights restored.")

    map_val = metrics["mAP50_95"]
    print(f"\nPer-channel smart mAP50-95 : {map_val:.4f}")
    print(f"FP32 baseline              : {FP32_MAP:.4f}  (delta {map_val - FP32_MAP:+.4f})")
    print(f"Per-tensor all             : {PERTENSOR_MAP:.4f}  (delta {map_val - PERTENSOR_MAP:+.4f})")
    print(f"Per-tensor smart (ex-DFL)  : {SMART_PT_MAP:.4f}  (delta {map_val - SMART_PT_MAP:+.4f})")

    out = save_eval_artifact(
        metrics,
        artifact_name="kernel_perchannel_eval",
        metadata={
            "model": "yolov8s",
            "method": "per_channel_int8_fake_quant",
            "quantized_layers": quant_layers,
            "skipped_layers": [DFL_LAYER],
            "dataset": "coco_val2017",
            "n_images": 5000,
            "fp32_baseline_mAP50_95": FP32_MAP,
            "pertensor_all_mAP50_95": PERTENSOR_MAP,
            "pertensor_smart_mAP50_95": SMART_PT_MAP,
        },
        env_json_path=ENV_JSON,
    )
    print(f"\nArtifact: {out}")


if __name__ == "__main__":
    main()
