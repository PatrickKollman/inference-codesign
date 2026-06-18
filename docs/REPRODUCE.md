# Reproducing Results

Step-by-step guide to reproduce every committed artifact in `results/` from a fresh
RunPod pod. Commands are in order; expected outputs and known failure modes are noted
at each step.

---

## Hardware and Software Requirements

| Component | Minimum | Used here |
|-----------|---------|-----------|
| GPU | RTX 3090 (CC 8.6) | RTX 4090 (CC 8.9) |
| CUDA | 12.x | 12.8 |
| PyTorch | 2.7+ | 2.8.0+cu128 |
| Python | 3.10+ | 3.12.3 |
| RAM | 16 GB | 64 GB (RunPod default) |
| Disk | ~5 GB | ~5 GB (COCO images + annotations + engines) |

**Latency numbers are hardware-specific.** The speedup ratios (3.61×, 4.60×) and dispatch
overhead fractions (37.8%) should reproduce qualitatively on any Ada Lovelace RTX card
with the same software stack, but exact ms values will differ on different GPUs.

---

## 1. Pod Setup

**RunPod template:** PyTorch 2.8 / CUDA 12.8 / Python 3.12 (Community Cloud).
Select an RTX 4090 instance. The template pre-installs PyTorch, CUDA, and cuDNN.

```bash
# On the pod, after SSH or browser terminal:
git clone https://github.com/PatrickKollman/inference-codesign.git
cd inference-codesign
bash scripts/setup_runpod.sh
```

`setup_runpod.sh` installs `ultralytics` and `nvidia-modelopt[torch]` (TRT bindings),
then runs `scripts/verify_env.py`, which writes `results/env.json` — the provenance
anchor for all downstream artifacts.

**Expected output from verify_env.py:**
```
GPU: NVIDIA GeForce RTX 4090 (CC 8.9)
CUDA: 12.8  PyTorch: 2.8.0+cu128  ultralytics: 8.4.68  Python: 3.12.3
Artifact: results/env.json
```

---

## 2. COCO val2017 Dataset

```bash
bash scripts/download_coco_val.sh
```

Downloads val2017 images (~778 MB) and annotations (~252 MB) from the official COCO
servers, extracts them to `data/coco/`, and converts COCO JSON annotations to YOLO
label format via `ultralytics.data.converter.convert_coco`.

**Expected output:**
```
5000 images extracted (expected 5000)
4952 label files created (expected ~4952; 48 val2017 images have no bbox annotations)
COCO val2017 ready at /path/to/inference-codesign/data/coco
```

**Known failure (ultralytics 8.4.70+):** The annotation converter crashes on
`captions_train2017.json` which lacks a `bbox` field. The download script works around
this by isolating `instances_val2017.json` in a temp directory before conversion.

**Known failure:** If `data/coco/labels/` already exists from a prior run, the move step
will fail with a path conflict. Delete `data/coco/labels/` and rerun.

---

## 3. FP32 Baseline

### Smoke test

```bash
python scripts/fp32_smoke_test.py
```

Single forward pass, confirms model loads and CUDA works. Writes `results/fp32_smoke_test.json`.
Run time: ~30 seconds (first run includes cuDNN algorithm selection).

### Latency benchmark

```bash
python scripts/fp32_benchmark.py
```

50 warmup + 200 timed iterations using CUDA event timing. Writes `results/fp32_latency.json`.

**Expected:** mean ~2.45 ms, p99 ~2.67 ms, std ~0.2 ms on RTX 4090.
Run time: ~2 minutes.

### Accuracy evaluation

```bash
python scripts/fp32_eval.py --data-dir data/coco
```

Full COCO val2017 (5000 images, batch=16). Writes `results/fp32_accuracy.json`.

**Expected:** mAP50-95 ≈ 0.4442, mAP50 ≈ 0.611.
Run time: ~15–20 minutes.

**Note on 44.9 discrepancy:** Ultralytics publishes 44.9 mAP50-95 for YOLOv8s. We
measure 44.42. **Isolation finding:** Stock `YOLO("yolov8s.pt").val()` with zero custom
parameters on ultralytics 8.4.70 gives the same 0.4442 — the gap is not in our harness.
It is upstream in the ultralytics version or checkpoint used to produce the published 44.9.
All deltas in this project are against our own baseline, so relative accuracy results are
unaffected.

---

## 4. Overhead Profiling

**Requires nsys 2025.x+ (pre-installed on RunPod PyTorch templates).**

```bash
nsys profile \
    --capture-range=cudaProfilerApi \
    --trace=cuda \
    --stats=true \
    --output=results/profile_nsys_timeline \
    --force-overwrite=true \
    python scripts/profile_nsys.py
```

50 warmup iterations run before `cudaProfilerStart()` — the timeline contains only
steady-state inference. 20 iterations captured.

**Expected output (nsys summary printed to terminal):**
```
CUDA GPU Kern Sum (20 iters): ~30.5 ms total → ~1.525 ms/iter
```

This 1.525 ms vs 2.453 ms measured wall-clock reveals the 0.928 ms dispatch gap.

**Nsight Compute (ncu) — blocked on Community Cloud:**

```bash
ncu --set=full --target-processes=all -o results/profile_ncu \
    python scripts/profile_nsys.py
# → Error: ERR_NVGPUCTRPERM — hardware counter access blocked
# RunPod Community Cloud containers lack SYS_ADMIN capability.
```

ncu was blocked. Per-kernel occupancy and memory throughput numbers in
`docs/profiling_analysis.md` are derived analytically (see that file's Methodology section).
To reproduce with real counters: use a RunPod Secure Cloud pod (supports `--privileged`)
or local hardware with `ncu` access.

---

## 5. Quantization Sensitivity Analysis

### Per-tensor INT8 baseline

```bash
python scripts/quant_pertensor_baseline.py --data-dir data/coco
```

Per-tensor symmetric INT8 on all 64 conv layers.
Writes `results/quant_pertensor_all.json`.

**Expected:** mAP50-95 ≈ 0.4399.
Run time: ~30 minutes.

### Per-layer sensitivity sweep

```bash
python scripts/quant_sensitivity_sweep.py --data-dir data/coco
```

Quantizes one layer at a time (all others FP32), evaluates on a 100-image COCO subset.
Identifies `model.22.dfl.conv` as the most sensitive layer — 16 parameters, uniquely
fragile because one shared scale collapses its learned bounding-box CDF.
Writes `results/quant_sensitivity.json`.

Run time: **~8 hours** (64 layers × ~7.5 min per 100-image eval). Run overnight.
Results are saved after each layer — safe to interrupt and inspect partial output.

---

## 6. Custom CUDA Kernel

### Kernel throughput benchmark

```bash
python scripts/kernel_benchmark.py
```

Benchmarks custom CUDA kernel vs PyTorch per-channel fake-quant across 6 representative
shapes. JIT-compiles `src/cuda/fake_quant_perchannel.cu` on first run (cached after).
Writes `results/kernel_benchmark.json`.

**Expected:** 4.9–6.1× speedup per shape. `[256,256,3,3]` at ~1624 GB/s
(exceeds DRAM peak — tensor fits in L2 cache).
Run time: ~5 minutes.

**Known issue:** JIT compilation takes 30–60 seconds on first run with no output. Not hanging.

### Per-channel accuracy evaluation

```bash
python scripts/kernel_eval.py --data-dir data/coco
```

Applies per-channel fake-quant (63/64 layers, DFL excluded) via the custom kernel,
runs full COCO val2017. Writes `results/kernel_perchannel_eval.json`.

**Expected:** mAP50-95 ≈ 0.4434.
Run time: **~45–90 minutes.**

**Known issue:** Progress bar may freeze for the first ~15 minutes before output. Not hung.

---

## 7. TensorRT Deployment

Verify TRT is installed: `python -c "import tensorrt as trt; print(trt.__version__)"` should
print `11.0.0.114` or newer. If not: `pip install "nvidia-modelopt[torch]"` (already run by
`setup_runpod.sh`).

### FP16 engine build

```bash
python scripts/trt_fp16_build.py
```

ONNX export → TRT engine build with `half=True`. No calibration data needed.
Writes `results/yolov8s_fp16.trt` (~23 MiB).
Run time: ~5–10 minutes.

### INT8 engine build

```bash
python scripts/trt_int8_build.py --data-dir data/coco
```

ONNX export → nvidia-modelopt INT8 calibration (512 COCO images) → TRT engine build.
Writes `results/yolov8s_int8.trt` (~11 MiB).
Run time: 5–15 minutes (calibration is the slow step).

**Known failure:** If the engine file already exists, the script exits immediately with
"Engine already exists — delete it to rebuild." Intentional guard against overwriting
committed artifacts.

### Latency benchmark

```bash
# FP16
python scripts/trt_benchmark.py --engine results/yolov8s_fp16.trt

# INT8
python scripts/trt_benchmark.py --engine results/yolov8s_int8.trt
```

50 warmup + 200 timed reps via direct TRT Python API. Writes `results/trt_fp16_latency.json`
and `results/trt_int8_latency.json`.

**Expected:** FP16 ≈ 0.679 ms (3.61×); INT8 ≈ 0.533 ms (4.60×).
Run time: ~2 minutes each.

### mAP evaluation

```bash
# FP16
python scripts/trt_eval.py --data-dir data/coco --engine results/yolov8s_fp16.trt

# INT8
python scripts/trt_eval.py --data-dir data/coco --engine results/yolov8s_int8.trt
```

Writes `results/trt_fp16_accuracy.json` and `results/trt_int8_accuracy.json`.

**Expected:** FP16 ≈ 0.4441 mAP (−0.0001); INT8 ≈ 0.4298 mAP (−0.0144).
Run time: ~15 minutes each.

---

## 8. Tests

```bash
# On pod (CUDA available):
pytest tests/ -v

# On local Mac (CPU/MPS only — no CUDA):
pytest tests/ -v
```

85 tests across four modules: harness timing/stats, eval metrics, INT8 fake-quant
correctness (reference), and CUDA kernel vs reference numerical match.

10 tests in `TestCudaMatchesReference` require a CUDA GPU — they will fail on a Mac
with no CUDA. All other 75 tests pass on both platforms.

Run time: ~30 seconds (CPU-only), ~36 seconds (CUDA, includes JIT compile on first run).

---

## 9. Portfolio Figures (No GPU Required)

```bash
# On local machine or pod — reads committed JSON artifacts, no GPU needed
pip install matplotlib numpy  # if not already installed
python scripts/make_figures.py
```

Writes `figures/fig1_pareto.png`, `fig2_profiling.png`, `fig3_kernel.png`, `fig4_sensitivity.png`.
All data is read from `results/*.json`; no inference is run.

---

## 10. Git Workflow Note

Both the pod and local machine have separate git histories during active development.
If you commit a result artifact on the pod while also working on the local machine,
the branches will diverge. Resolve with:

```bash
# On local machine after pod pushes a new commit:
git pull --rebase
git push
```

Do not use `git pull --merge` — the linear history is cleaner and easier to audit
for a measurement-only commit sequence.

---

## Expected Artifacts

| Artifact | Measurement | Expected value |
|----------|-------------|----------------|
| `env.json` | Environment provenance | timestamp_utc, GPU, CUDA, PyTorch versions |
| `fp32_latency.json` | FP32 latency | mean_ms ≈ 2.453, p99_ms ≈ 2.668 |
| `fp32_accuracy.json` | FP32 accuracy | mAP50_95 ≈ 0.4442 |
| `profile_nsys_stats.txt` | Nsight kernel summary | total kernel time ~30.5 ms / 20 iters |
| `quant_pertensor_all.json` | Per-tensor INT8 mAP | mAP50_95 ≈ 0.4399 |
| `quant_pertensor_smart.json` | Per-tensor smart mAP | mAP50_95 ≈ 0.4425 |
| `quant_sensitivity.json` | Per-layer sensitivity | 64 layers, DFL drop ≈ +0.0092 |
| `kernel_benchmark.json` | Kernel speedup by shape | 4.9–6.1× per shape |
| `kernel_perchannel_eval.json` | Per-channel mAP | mAP50_95 ≈ 0.4434 |
| `trt_fp16_latency.json` | TRT FP16 latency | mean_ms ≈ 0.679 |
| `trt_fp16_accuracy.json` | TRT FP16 accuracy | mAP50_95 ≈ 0.4441 |
| `trt_int8_latency.json` | TRT INT8 latency | mean_ms ≈ 0.533 |
| `trt_int8_accuracy.json` | TRT INT8 accuracy | mAP50_95 ≈ 0.4298 |
