"""COCO mAP evaluation utilities.

Accuracy baseline uses ultralytics YOLO.val() — it owns preprocessing,
postprocessing, and NMS for the accuracy measurement. This is intentional:
mAP is a property of the model weights, not the inference pipeline. Using
the well-tested ultralytics path avoids reinventing decode+NMS and ensures
the same eval pipeline is used for FP32, INT8, and AMP comparisons.

Latency measurement is handled separately by src/harness.py.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent.parent / "results"

# COCO 80-class names in index order
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _make_coco_yaml(data_dir: Path, val_split: str = "images/val2017") -> str:
    """Generate a minimal ultralytics-compatible COCO yaml string."""
    names_lines = "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO_CLASSES))
    return (
        f"path: {data_dir.resolve()}\n"
        f"train: images/val2017\n"  # required by ultralytics validator; unused during val
        f"val: {val_split}\n"
        f"nc: 80\n"
        f"names:\n{names_lines}\n"
    )


def make_subset_txt(data_dir: Path, n_images: int, seed: int = 42) -> Path:
    """Write a deterministic subset of val2017 image paths to a .txt file.

    ultralytics accepts a text file of relative image paths as the val split.
    The file is written inside data_dir and is safe to call repeatedly
    (same seed → same file, idempotent).

    Returns the path to the .txt file (relative use: val: subset_{n}.txt).
    """
    import random

    images = sorted((data_dir / "images" / "val2017").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No val2017 images found in {data_dir}/images/val2017/")

    rng = random.Random(seed)
    subset = sorted(rng.sample(images, min(n_images, len(images))))

    txt_path = data_dir / f"subset_{n_images}.txt"
    with open(txt_path, "w") as f:
        for img in subset:
            # Write paths relative to data_dir so ultralytics resolves correctly
            f.write(f"./images/val2017/{img.name}\n")

    return txt_path


def run_coco_val(
    weights: str | Path,
    data_dir: str | Path,
    device: str = "cuda",
    img_sz: int = 640,
    batch: int = 16,
    workers: int = 4,
    n_images: int | None = None,
) -> Any:
    """Run ultralytics YOLO.val() on COCO val2017 and return the raw result.

    Writes a temporary yaml (machine-specific paths can't be committed) and
    cleans it up regardless of success or failure.

    Args:
        weights:  Path to .pt file or name for auto-download (e.g. "yolov8s.pt").
        data_dir: Root of the COCO dataset (expects images/val2017/ and
                  annotations/instances_val2017.json inside).
        device:   "cuda" or "cpu".
        img_sz:   Inference resolution. Must match the latency harness.
        batch:    Images per batch. Doesn't affect mAP; tune for GPU memory.
        workers:  DataLoader workers.

    Returns:
        Ultralytics validator result object. Pass to parse_metrics().
    """
    from ultralytics import YOLO

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"COCO data directory not found: {data_dir}\n"
            "Run scripts/download_coco_val.sh first."
        )

    if n_images is not None:
        subset_txt = make_subset_txt(data_dir, n_images)
        # Path relative to data_dir root for the yaml val key
        val_split = f"./{subset_txt.name}"
        yaml_content = _make_coco_yaml(data_dir, val_split=val_split)
    else:
        yaml_content = _make_coco_yaml(data_dir)

    tmp_fd, tmp_yaml = tempfile.mkstemp(suffix=".yaml", prefix="coco_val_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(yaml_content)
        model = YOLO(str(weights))
        return model.val(
            data=tmp_yaml,
            imgsz=img_sz,
            batch=batch,
            device=device,
            workers=workers,
            verbose=False,
            save=False,
            save_json=False,
        )
    finally:
        os.unlink(tmp_yaml)


def parse_metrics(val_result: Any) -> dict:
    """Extract mAP metrics from an ultralytics val() result object.

    Tries results_dict first (stable across versions), falls back to
    box.* attributes. Returns None for any metric that can't be found
    rather than raising — missing metrics are a finding, not a crash.
    """
    rd = getattr(val_result, "results_dict", {})

    def _get(results_dict_key: str, attr_path: str) -> float | None:
        if results_dict_key in rd:
            return float(rd[results_dict_key])
        obj = val_result
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        try:
            return float(obj)
        except (TypeError, ValueError):
            return None

    return {
        "mAP50_95": _get("metrics/mAP50-95(B)", "box.map"),
        "mAP50":    _get("metrics/mAP50(B)",    "box.map50"),
        "precision": _get("metrics/precision(B)", "box.mp"),
        "recall":    _get("metrics/recall(B)",    "box.mr"),
    }


def save_eval_artifact(
    metrics: dict,
    artifact_name: str,
    metadata: dict | None = None,
    env_json_path: Path | None = None,
) -> Path:
    """Write eval artifact to results/{artifact_name}.json.

    Mirrors the provenance convention from save_timing_artifact() — every
    accuracy artifact links back to the verified env timestamp.
    """
    env_timestamp = None
    if env_json_path is not None and env_json_path.exists():
        with open(env_json_path) as f:
            env_timestamp = json.load(f).get("timestamp_utc")
    elif env_json_path is not None:
        env_timestamp = f"[MISSING: {env_json_path}]"

    artifact = {
        "env_timestamp_utc": env_timestamp,
        **(metadata or {}),
        "metrics": metrics,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{artifact_name}.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    return out_path
