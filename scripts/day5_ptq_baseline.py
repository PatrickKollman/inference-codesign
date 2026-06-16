"""Day 5: full-model INT8 weight fake-quantization accuracy baseline.

Applies fake INT8 quantization to ALL Conv2d weights simultaneously,
evaluates mAP on full COCO val2017, and records the result.

This gives the accuracy floor for "quantize everything" — the lower bound
of the Pareto. Configs that skip sensitive layers will sit above this floor.

Note on GPU INT8 latency
------------------------
Fake quantization runs FP32 compute on INT8-range weights — it measures
accuracy impact, not latency impact. GPU INT8 tensor core speedup requires
a compiler-path deployment (TensorRT) covered in Layer 3. The latency axis
of the Day 5 Pareto is estimated from the roofline analysis in docs/profile_notes.md
until Layer 3 provides measured numbers.

Usage:
    python scripts/day5_ptq_baseline.py [--data-dir data/coco]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.eval import parse_metrics, run_coco_val, save_eval_artifact
from src.quantize import apply_weight_fake_quant, iter_conv_modules, restore_weights

ENV_JSON = ROOT / "results" / "day1_env.json"
WEIGHTS = "yolov8s.pt"
DEVICE = "cuda"
IMG_SZ = 640


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "coco"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    from ultralytics import YOLO
    yolo = YOLO(WEIGHTS)
    model = yolo.model
    model.eval()

    n_conv = sum(1 for _ in iter_conv_modules(model))
    print(f"Conv2d layers to quantize: {n_conv}")

    # Apply fake INT8 to all conv weights
    originals = apply_weight_fake_quant(model)
    print("All Conv2d weights fake-quantized to INT8 range.")

    # Evaluate on full val5000
    print("Running COCO val2017 (5000 images)...")
    result = yolo.val(
        data=_make_tmp_yaml(data_dir),
        imgsz=IMG_SZ,
        device=DEVICE,
        batch=16,
        workers=4,
        verbose=False,
        save=False,
        save_json=False,
    )

    restore_weights(model, originals)

    metrics = parse_metrics(result)
    print("\nResults (INT8 weight fake-quant, all conv layers):")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}" if v is not None else f"  {k:15s}: N/A")

    metadata = {
        "model": "yolov8s",
        "quantization": "per-tensor symmetric INT8 weight fake-quant (all conv)",
        "n_conv_quantized": n_conv,
        "dataset": "coco_val2017",
        "n_images": 5000,
        "img_sz": IMG_SZ,
        "device": DEVICE,
        "note": "FP32 compute, INT8-range weights. Accuracy measurement only.",
    }
    out = save_eval_artifact(
        metrics,
        artifact_name="day5_ptq_baseline",
        metadata=metadata,
        env_json_path=ENV_JSON,
    )
    print(f"\nArtifact: {out}")


def _make_tmp_yaml(data_dir: Path) -> str:
    """Write a temp coco yaml and return its path. Caller owns cleanup."""
    import os
    import tempfile
    from src.eval import _make_coco_yaml
    yaml_content = _make_coco_yaml(data_dir)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="coco_ptq_")
    with os.fdopen(tmp_fd, "w") as f:
        f.write(yaml_content)
    return tmp_path


if __name__ == "__main__":
    main()
