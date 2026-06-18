#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod for inference-codesign.
# Assumes: PyTorch 2.x / CUDA 12.x template (CUDA and torch pre-installed).
# Run once after cloning the repo on the pod.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== inference-codesign RunPod setup ==="
echo "Repo root: ${REPO_ROOT}"

# Install Python dependencies
pip install --upgrade pip
pip install ultralytics
pip install ninja  # required for torch.utils.cpp_extension JIT (CUDA kernel tests)

# TensorRT (~5 min install)
# nvidia-modelopt pulls TRT Python bindings as a dependency.
pip install "nvidia-modelopt[torch]"

# Verify environment and write provenance artifacts
echo ""
echo "=== Verifying environment ==="
python "${REPO_ROOT}/scripts/verify_env.py"

echo ""
echo "=== Setup complete. Run FP32 smoke test: ==="
echo "  python scripts/fp32_smoke_test.py"
