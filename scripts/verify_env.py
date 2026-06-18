"""Collect environment facts and write results/env.json.

Run this first on any new pod before any benchmark.
Output files are the provenance anchor for all downstream reported numbers.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip()
    except subprocess.CalledProcessError as e:
        return f"[ERROR] {e.output.strip()}"


def main() -> None:
    env: dict = {}
    env["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    env["nvidia_smi"] = run("nvidia-smi")
    env["nvcc_version"] = run("nvcc --version")
    env["python_version"] = sys.version

    try:
        import torch
        env["torch_version"] = torch.__version__
        env["torch_cuda_version"] = torch.version.cuda
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
            env["cuda_device_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            env["cuda_compute_capability"] = f"{props.major}.{props.minor}"
            env["cuda_total_memory_gb"] = round(props.total_memory / 1e9, 2)
            env["cudnn_version"] = str(torch.backends.cudnn.version())
        else:
            env["cudnn_version"] = "N/A (no CUDA)"
    except ImportError:
        env["torch_version"] = "NOT INSTALLED"

    try:
        import ultralytics
        env["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        env["ultralytics_version"] = "NOT INSTALLED"

    # Write raw JSON artifact first
    RESULTS.mkdir(exist_ok=True)
    json_path = RESULTS / "env.json"
    with open(json_path, "w") as f:
        json.dump(env, f, indent=2)

    gpu_name = env.get("cuda_device_name", "N/A")
    cc = env.get("cuda_compute_capability", "N/A")
    vram = env.get("cuda_total_memory_gb", "N/A")

    print(f"[verify_env] JSON artifact: {json_path}")
    print(f"[verify_env] Device: {gpu_name}  CC: {cc}  VRAM: {vram} GB")
    print(f"[verify_env] PyTorch {env.get('torch_version')}  CUDA {env.get('torch_cuda_version')}  cuDNN {env.get('cudnn_version')}")


if __name__ == "__main__":
    main()
