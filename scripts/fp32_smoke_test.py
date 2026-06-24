"""Verify YOLOv8s forward pass and commit output-shape artifact.

This is a smoke test only — not a benchmark. It confirms:
  - The model graph is accessible via plain nn.Module forward()
  - Output shapes are known before harness construction begins
  - No ultralytics predict() wrapper is in the path

Run after verify_env.py.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import load_yolov8s, parameter_count

RESULTS = ROOT / "results"
INPUT_SHAPE = (1, 3, 640, 640)  # standard COCO inference resolution


def tensor_shape(x) -> list:
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    return [type(x).__name__]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke_test] Device: {device}")

    model = load_yolov8s(device=device)
    n_params = parameter_count(model)
    print(f"[smoke_test] Parameters: {n_params:,}")

    x = torch.zeros(INPUT_SHAPE, device=device, dtype=torch.float32)
    print(f"[smoke_test] Input shape: {list(INPUT_SHAPE)}  dtype: float32")

    with torch.no_grad():
        out = model(x)

    # Capture output structure without assuming shape
    if isinstance(out, (list, tuple)):
        output_info = [tensor_shape(o) for o in out]
    else:
        output_info = [tensor_shape(out)]

    print(f"[smoke_test] Output (list of shapes): {output_info}")

    # Env reference for provenance
    env_path = RESULTS / "env.json"
    env_timestamp = None
    if env_path.exists():
        with open(env_path) as f:
            env_timestamp = json.load(f).get("timestamp_utc")
    else:
        print("[smoke_test] WARNING: env.json not found — run verify_env.py first")

    artifact = {
        "env_timestamp_utc": env_timestamp,
        "model": "yolov8s",
        "weights": "yolov8s.pt",
        "input_shape": list(INPUT_SHAPE),
        "input_dtype": "float32",
        "n_parameters": n_params,
        "output_shapes": output_info,
        "device": device,
        "inference_wrapper": "none (plain nn.Module forward())",
    }

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / "fp32_smoke_test.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[smoke_test] Artifact written: {out_path}")


if __name__ == "__main__":
    main()
