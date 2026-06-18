"""Day 3: FP32 baseline mAP evaluation on COCO val2017.

Produces results/fp32_accuracy.json — the accuracy anchor for all downstream
quantization comparisons. Run after Day 1 and Day 2 artifacts are committed.

Usage:
    python scripts/day3_eval_fp32.py --data-dir data/coco
    python scripts/day3_eval_fp32.py --data-dir /absolute/path/to/coco
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.eval import parse_metrics, run_coco_val, save_eval_artifact

ENV_JSON = ROOT / "results" / "env.json"
WEIGHTS = "yolov8s.pt"
DEVICE = "cuda"
IMG_SZ = 640   # must match harness INPUT_SHAPE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "coco"),
        help="Root of COCO dataset (contains images/val2017/ and annotations/).",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print(f"Model:    {WEIGHTS}")
    print(f"Device:   {DEVICE}")
    print(f"Data dir: {data_dir}")
    print(f"Img size: {IMG_SZ}")
    print()

    val_result = run_coco_val(
        weights=WEIGHTS,
        data_dir=data_dir,
        device=DEVICE,
        img_sz=IMG_SZ,
        batch=16,
        workers=4,
    )

    metrics = parse_metrics(val_result)

    print("Results:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}" if v is not None else f"  {k:15s}: N/A")

    metadata = {
        "model": "yolov8s",
        "precision": "fp32",
        "dataset": "coco_val2017",
        "n_images": 5000,
        "img_sz": IMG_SZ,
        "batch": 16,
        "device": DEVICE,
    }
    out = save_eval_artifact(
        metrics,
        artifact_name="fp32_accuracy",
        metadata=metadata,
        env_json_path=ENV_JSON,
    )
    print(f"\nArtifact: {out}")


if __name__ == "__main__":
    main()
