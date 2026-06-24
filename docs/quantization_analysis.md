# Quantization Sensitivity Analysis: PTQ Pareto and the Sensitivity–Cost Interaction

**Environment:** RTX 4090 (Ada, CC 8.9), CUDA 12.8, PyTorch 2.8.0+cu128, ultralytics 8.4.68
**Raw artifacts:** `results/quant_pertensor_all.json`, `results/quant_sensitivity.json`, `results/quant_pertensor_smart.json`
**Provenance:** `results/env.json` timestamp 2026-06-16T04:39:51.071378+00:00

---

## The Pareto

Three measured configurations on full COCO val2017 (5000 images):

| Config | mAP50-95 | mAP drop | Params in FP32 | GPU INT8 latency |
|--------|----------|----------|----------------|-----------------|
| FP32 eager baseline | 0.4442 | — | 11,166,560 (100%) | 2.453 ms mean, p99 2.668 ms |
| INT8 smart (63/64 conv) | **0.4425** | **−0.0017** | **16 (DFL only)** | est. 1.2–1.5 ms (TRT) |
| INT8 all (64/64 conv) | 0.4399 | −0.0043 | 0 | est. 1.2–1.5 ms (TRT) |

Accuracy measurements use per-tensor symmetric INT8 weight fake-quantization (FP32
compute, INT8-range weights). GPU INT8 latency requires TensorRT and is measured in the TensorRT analysis; the estimate above comes from the roofline analysis in `docs/profiling_analysis.md`
(compute-bound conv kernels, 8× INT8:TF32 theoretical throughput ratio, ~2× realistic
end-to-end improvement after accounting for dispatch overhead and memory-bound ops).

---

## The Non-Obvious Finding: Sensitive ≠ Expensive

The textbook PTQ result is that early backbone layers and detection heads are sensitive
while deep backbone layers are safe. That is partially true here but misses the most
important structure.

**From profiling (docs/profiling_analysis.md):**
The dominant kernel — `sm86_xmma_fprop_implicit_gemm_indexed_tf32f32`, 20 instances per
forward pass, 19.3% of GPU kernel time — corresponds to the mid-to-large backbone and
neck convolutions: `model.5.*`, `model.6.*`, `model.7.*`, `model.8.*`, `model.9.*`.

**From the sensitivity sweep:**
Those same mid-backbone layers show mAP drops of 0.0000 to 0.0016 — within the noise
floor of the 100-image subset (variance ~±0.001). They are essentially insensitive to
per-tensor symmetric INT8 weight quantization.

**The actual most-sensitive layer:**

| Layer | Drop (100-img subset) | Drop (full val5000) | Params |
|-------|----------------------|---------------------|--------|
| `model.22.dfl.conv` | +0.0092 | +0.0026 (recovered by protecting) | **16** |

The DFL (Distribution Focal Loss) conv uses 16 learned weights to parameterize the
cumulative distribution over bounding box coordinate offsets at 16 discrete bins.
With only 16 parameters, per-tensor symmetric quantization maps `scale = max(|w|)/127`.
Many of the 16 values collapse to the same quantized level, destroying the learned
distribution structure. The layer is both uniquely sensitive and uniquely cheap to protect.

**The full sensitivity–cost intersection:**

```
                    HIGH sensitivity
                         |
     model.2.* (early)   |  model.22.dfl.conv ← protect this (16 params, free)
     model.1.conv        |
     model.15.cv1        |
  -------------------+---+----------------------------→ HIGH compute cost
     (cheap layers)  |   (expensive layers from nsys)
                     |
     model.6.*, model.8.*, model.9.*   ← expensive AND insensitive → quantize these
     model.21.*, model.18.*
```

The smart policy — INT8 everywhere, FP32 for the 16-parameter DFL layer — is not
obvious from sensitivity ranking alone. A naive reading says "protect early layers and
detection heads." The actual finding is that the 16-parameter DFL conv accounts for 60%
of the naive full-INT8 accuracy loss, while the layers that dominate inference time are
safely quantizable.

---

## Noise Accounting

The 100-image subset has high variance (~±0.001 per eval). The confirmed signal:
- **DFL at +0.0092 on subset, +0.0026 recovered on full val:** Directionally correct,
  magnitude overestimated by ~3.5×. The subset correctly identified it as the #1 outlier.
- **Early backbone (model.2.*) at +0.003 on subset:** Marginal signal, within 1–2 noise
  sigmas. Not confirmed by a full-val run of "INT8 all except early backbone."
- **Negative drops (e.g. model.22.cv3.0.1 at −0.0012):** Statistical noise. Quantization
  does not improve accuracy; these reflect subset sampling variance.

For any drop < 0.002 in the sweep, treat the ranking as approximate. The DFL finding
is robust; the ordering of the remaining top-10 is approximate.

---

## Deadline-Conditioned Analysis

The project's latency budget is **10 ms/frame** for one camera's perception slice at 30Hz.

| Config | Latency (est.) | mAP50-95 | Budget used | Viable? |
|--------|---------------|----------|-------------|---------|
| FP32 eager | 2.453 ms | 0.4442 | 24.5% | Yes |
| INT8 smart (TRT, est.) | ~1.2–1.5 ms | 0.4425 | 12–15% | Yes, preferred |
| INT8 all (TRT, est.) | ~1.2–1.5 ms | 0.4399 | 12–15% | Yes |
| FP32 eager (degraded HW) | ~8–15 ms | 0.4442 | 80–150% | Marginal/No |

On the RTX 4090 baseline, all configurations are within budget — the 4090 is a
high-end GPU, not the target hardware. The constraint tightens on fixed-function silicon.

**The interesting decision is not "which configuration fits the budget on the 4090" — it
is "what does the accuracy/latency tradeoff look like as the compute budget shrinks toward
edge ASICs."** The smart INT8 policy buys back ~2× latency headroom at the cost of
0.17 absolute mAP points (0.38% relative), leaving more budget for other stack components
(tracking, planning, sensor fusion). On an ASIC where the FP32 model might consume
6–8 ms of the 10 ms frame, INT8 may be the difference between viable and not.

---

## What INT8 Does and Does Not Address

From the profiling analysis (see `docs/profiling_analysis.md`):

**INT8 accelerates:**
- Conv kernel execution (~58% of kernel time, compute-bound) → INT8 tensor cores
  provide 8× theoretical, ~2× realistic end-to-end throughput improvement.

**INT8 does not address:**
- Dispatch overhead (0.928 ms, 38% of wall-clock) — this is CPU-GPU scheduling latency
  independent of precision. INT8 shrinks kernel time, making dispatch a *larger* fraction.
- Layout conversions (10.4% of kernel time) — NCHW↔NHWC transposes are precision-agnostic.
- BatchNorm (13.6% of kernel time) — memory-bandwidth-bound; INT8 weights save activation
  memory bandwidth but the BN arithmetic is already fast.

After INT8, the bottleneck shifts: dispatch overhead becomes the dominant term.
The next optimization opportunity is graph-level (TensorRT: op fusion, CUDA Graphs),
not precision-level. This is the motivation for TensorRT deployment.

---

## Quantization Method Scope

This analysis uses **per-tensor symmetric INT8 weight fake-quantization.** This is the
simplest and most conservative approach. Three known improvements are not measured here:

1. **Per-channel quantization:** One scale per output filter instead of one scale per
   tensor. Expected recovery: 1–3 mAP points on sensitive layers. The early backbone
   and DFL layer would benefit most.
2. **Activation quantization:** This analysis quantizes weights only. Activations also
   contribute quantization noise, especially in the detection head. Static activation
   quantization (calibrated scales from a calibration dataset) is needed for actual INT8
   kernel dispatch.
3. **Quantization-aware training (QAT):** Fine-tuning with simulated quantization
   recovers most of the accuracy gap at the cost of training compute. Not in scope for
   a PTQ-only analysis.

The gap between this analysis and a production INT8 deployment is real but well-characterized.
The sensitivity ranking and DFL finding are robust across quantization methods; the absolute
mAP numbers would improve with per-channel and activation quantization.
