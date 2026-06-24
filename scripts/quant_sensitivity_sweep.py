"""Per-layer INT8 weight sensitivity sweep.

For each Conv2d in YOLOv8s, temporarily fake-quantizes its weights to INT8
precision (per-tensor symmetric), evaluates mAP on a small COCO subset, and
records the mAP drop vs FP32 baseline.

Strategy
--------
* Weight-only fake quantization (per-tensor symmetric): worst-case sensitivity.
  Per-channel quantization recovers 1-3 mAP points; we use per-tensor here to
  amplify the signal and identify clearly sensitive vs clearly insensitive layers.
* Fast subset (~100 images) for the sweep — enough for relative ranking, not for
  absolute mAP citation. Only the final Pareto configs run on full val5000.
* Results saved after each layer so the sweep is resumable on interruption.
* Uses yolo.val() on the already-loaded model so in-memory weight modifications
  are picked up directly — NOT run_coco_val() which reloads from disk.

Cross-reference with profiling results (results/profile_nsys_stats.txt) to find
the intersection of expensive layers (from nsys) and sensitive layers (from here).

Usage:
    python scripts/quant_sensitivity_sweep.py [--data-dir data/coco] [--n-images 100]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.eval import COCO_CLASSES, make_subset_txt, parse_metrics
from src.quantize import (
    count_conv_params,
    fake_quantize_int8_symmetric,
    iter_conv_modules,
)

ENV_JSON = ROOT / "results" / "env.json"
WEIGHTS = "yolov8s.pt"
DEVICE = "cuda"
IMG_SZ = 640


def _make_yaml_for_subset(data_dir: Path, subset_txt: Path) -> str:
    names_lines = "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO_CLASSES))
    return (
        f"path: {data_dir.resolve()}\n"
        f"train: images/val2017\n"
        f"val: ./{subset_txt.name}\n"
        f"nc: 80\n"
        f"names:\n{names_lines}\n"
    )


def val_with_model(yolo, yaml_path: str, device: str, img_sz: int) -> dict:
    """Run yolo.val() on the already-loaded (possibly modified) yolo.model."""
    result = yolo.val(
        data=yaml_path,
        imgsz=img_sz,
        device=device,
        batch=16,
        workers=4,
        verbose=False,
        save=False,
        save_json=False,
    )
    return parse_metrics(result)


def _read_env_timestamp() -> str | None:
    if ENV_JSON.exists():
        with open(ENV_JSON) as f:
            return json.load(f).get("timestamp_utc")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "coco"))
    parser.add_argument(
        "--n-images", type=int, default=100,
        help="Subset size for fast sweep. Use 500+ for higher-fidelity results.",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    from ultralytics import YOLO
    yolo = YOLO(WEIGHTS)
    model = yolo.model
    model.eval()

    # Write subset txt once; reuse for all eval calls
    subset_txt = make_subset_txt(data_dir, args.n_images)
    yaml_content = _make_yaml_for_subset(data_dir, subset_txt)

    tmp_fd, tmp_yaml = tempfile.mkstemp(suffix=".yaml", prefix="coco_sweep_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(yaml_content)
        _run_sweep(yolo, model, data_dir, tmp_yaml, args.n_images)
    finally:
        os.unlink(tmp_yaml)


def _run_sweep(yolo, model, data_dir: Path, yaml_path: str, n_images: int) -> None:
    # --- FP32 baseline on subset ---
    print(f"Evaluating FP32 baseline on {n_images}-image subset...")
    t0 = time.time()
    baseline_metrics = val_with_model(yolo, yaml_path, DEVICE, IMG_SZ)
    baseline_mAP = baseline_metrics["mAP50_95"] or 0.0
    print(f"  FP32 baseline mAP50-95: {baseline_mAP:.4f}  ({time.time()-t0:.0f}s)")

    # --- Per-layer sweep ---
    conv_layers = list(iter_conv_modules(model))
    print(f"\nSweeping {len(conv_layers)} Conv2d layers ({n_images} images each)...")
    print(f"Estimated time: ~{len(conv_layers) * 15 // 60}–{len(conv_layers) * 30 // 60} min\n")

    out_path = ROOT / "results" / "quant_sensitivity.json"
    results = []

    for i, (name, conv) in enumerate(conv_layers):
        n_params = count_conv_params(conv)
        original_weight = conv.weight.data.clone()

        conv.weight.data = fake_quantize_int8_symmetric(conv.weight.data)

        t0 = time.time()
        metrics = val_with_model(yolo, yaml_path, DEVICE, IMG_SZ)
        elapsed = time.time() - t0

        mAP = metrics["mAP50_95"] if metrics["mAP50_95"] is not None else 0.0
        drop = baseline_mAP - mAP

        conv.weight.data = original_weight  # restore before next iteration

        entry = {
            "layer": name,
            "n_params": n_params,
            "mAP50_95": round(mAP, 4),
            "mAP_drop": round(drop, 4),
        }
        results.append(entry)

        print(f"[{i+1:3d}/{len(conv_layers)}] {name:<55s}  "
              f"drop={drop:+.4f}  params={n_params:,}  ({elapsed:.0f}s)")

        # Save after each layer — allows resuming on interruption
        with open(out_path, "w") as f:
            json.dump({
                "env_timestamp_utc": _read_env_timestamp(),
                "n_images_subset": n_images,
                "baseline_mAP50_95": round(baseline_mAP, 4),
                "quantization": "per-tensor symmetric INT8 (weight-only fake quant)",
                "note": "mAP values from fast subset — use for relative ranking only",
                "results": results,
            }, f, indent=2)

    # Final summary
    ranked = sorted(results, key=lambda r: r["mAP_drop"], reverse=True)
    print(f"\n--- Top 10 most sensitive (protect from INT8) ---")
    for r in ranked[:10]:
        print(f"  {r['layer']:<55s}  drop={r['mAP_drop']:+.4f}  params={r['n_params']:,}")

    print(f"\n--- Top 10 least sensitive (safe to quantize) ---")
    for r in ranked[-10:]:
        print(f"  {r['layer']:<55s}  drop={r['mAP_drop']:+.4f}  params={r['n_params']:,}")

    print(f"\nArtifact: {out_path}")


if __name__ == "__main__":
    main()
