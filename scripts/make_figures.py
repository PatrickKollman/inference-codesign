"""Generate portfolio figures from committed JSON artifacts.

Reads:  results/*.json, results/profile_nsys_stats.txt
Writes: figures/fig1_pareto.png
        figures/fig2_profiling.png
        figures/fig3_kernel.png
        figures/fig4_sensitivity.png

Usage:
    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# ── Palette ──────────────────────────────────────────────────────────────────
C = {
    "fp32":           "#6b7280",   # gray
    "pertensor_all":  "#ef4444",   # red
    "pertensor_smart":"#f97316",   # orange
    "perchannel":     "#3b82f6",   # blue
    "trt_fp16":       "#8b5cf6",   # purple
    "trt_int8":       "#10b981",   # emerald
    "dispatch":       "#ef4444",
    "conv":           "#3b82f6",
    "bn":             "#f97316",
    "layout":         "#f59e0b",
    "silu":           "#10b981",
    "other":          "#d1d5db",
    "pytorch":        "#9ca3af",
    "cuda":           "#3b82f6",
}

STYLE = {
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         150,
}
plt.rcParams.update(STYLE)


def load(name: str) -> dict:
    with open(RESULTS / name) as f:
        return json.load(f)


# ── Figure 1: Latency–Accuracy Pareto ────────────────────────────────────────

def fig1_pareto():
    fp32   = load("fp32_latency.json")
    fp32_m = load("fp32_accuracy.json")
    fp16_b = load("trt_fp16_latency.json")
    fp16_e = load("trt_fp16_accuracy.json")
    int8_b = load("trt_int8_latency.json")
    int8_e = load("trt_int8_accuracy.json")
    pc_e   = load("kernel_perchannel_eval.json")
    sp_e   = load("quant_pertensor_smart.json")
    pt_all = load("quant_pertensor_all.json")

    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.subplots_adjust(bottom=0.23)

    fp32_lat = fp32["timing"]["mean_ms"]  # 2.453

    # ── Inference configurations (actual deployment points) ──────────────────
    inf_points = [
        ("FP32 Eager",   fp32_lat,                    fp32["timing"]["p99_ms"],   fp32_m["metrics"]["mAP50_95"], C["fp32"],    "o", 140),
        ("TRT FP16",     fp16_b["timing"]["mean_ms"],  fp16_b["timing"]["p99_ms"], fp16_e["metrics"]["mAP50_95"],C["trt_fp16"],"^", 140),
        ("TRT INT8",     int8_b["timing"]["mean_ms"],  int8_b["timing"]["p99_ms"], int8_e["metrics"]["mAP50_95"],C["trt_int8"],"s", 140),
    ]

    # Draw Pareto frontier line
    lat_f = [p[1] for p in inf_points]
    map_f = [p[3] for p in inf_points]
    ax.plot(lat_f, map_f, "--", color="#9ca3af", lw=1.2, zorder=1, label="_nolegend_")

    # Plot error bars (mean to p99) for inference points
    for label, mean_ms, p99_ms, mAP, color, marker, ms in inf_points:
        ax.errorbar(mean_ms, mAP, xerr=[[0], [p99_ms - mean_ms]],
                    fmt="none", color=color, capsize=3, lw=1.2, zorder=2)
        ax.scatter(mean_ms, mAP, color=color, marker=marker, s=ms,
                   zorder=4, edgecolors="white", linewidths=1.0)

    # ── Fake-quant points: jitter x slightly so they don't stack ─────────────
    fq_jitter  = [-0.06, 0.0, 0.06]   # small x offset for each
    fq_points  = [
        ("Per-tensor all (64/64)",   pt_all["metrics"]["mAP50_95"], C["pertensor_all"],  "D", 70),
        ("Per-tensor smart (–DFL)",  sp_e["metrics"]["mAP50_95"],   C["pertensor_smart"],"D", 70),
        ("Per-channel smart (–DFL)", pc_e["metrics"]["mAP50_95"],  C["perchannel"],      "D", 70),
    ]

    for (label, mAP, color, marker, ms), jitter in zip(fq_points, fq_jitter):
        ax.scatter(fp32_lat + jitter, mAP, color=color, marker=marker, s=ms,
                   zorder=3, edgecolors="white", linewidths=0.8, alpha=0.9)

    # ── Annotations: inference points ────────────────────────────────────────
    inf_ann = [
        # (label, mean_ms, mAP, color, text_xy, ha)
        ("FP32 Eager\n(baseline)",   fp32_lat,                    fp32_m["metrics"]["mAP50_95"],  C["fp32"],    (fp32_lat - 0.22, 0.4454), "right"),
        ("TRT FP16\n3.61× faster",   fp16_b["timing"]["mean_ms"], fp16_e["metrics"]["mAP50_95"],  C["trt_fp16"],(fp16_b["timing"]["mean_ms"] + 0.12, 0.4454), "left"),
        ("TRT INT8\n4.60× faster",   int8_b["timing"]["mean_ms"], int8_e["metrics"]["mAP50_95"],  C["trt_int8"],(int8_b["timing"]["mean_ms"] + 0.08, 0.4290), "left"),
    ]
    for label, px, py, color, (tx, ty), ha in inf_ann:
        ax.annotate(label, xy=(px, py), xytext=(tx, ty),
                    fontsize=8, color=color, ha=ha, va="center", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7))

    # ── Annotations: fake-quant points (right side, stacked vertically) ──────
    fq_ann_x = fp32_lat + 0.25   # text column to the right of the cluster
    for (label, mAP, color, _, _), jitter in zip(fq_points, fq_jitter):
        ax.annotate(label, xy=(fp32_lat + jitter, mAP),
                    xytext=(fq_ann_x, mAP),
                    fontsize=7.5, color=color, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.5))

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=C["fp32"],           label="FP32 Eager — inference baseline"),
        mpatches.Patch(color=C["trt_fp16"],        label="TRT FP16 — compiled engine + CUDA Graphs"),
        mpatches.Patch(color=C["trt_int8"],        label="TRT INT8 — compiled engine + CUDA Graphs"),
        mpatches.Patch(color=C["pertensor_all"],   label="Fake-quant: per-tensor all (64/64)"),
        mpatches.Patch(color=C["pertensor_smart"], label="Fake-quant: per-tensor smart (63/64)"),
        mpatches.Patch(color=C["perchannel"],      label="Fake-quant: per-channel smart (63/64)"),
    ]
    # Place legend below the axes to guarantee no overlap with any data point.
    ax.legend(handles=handles, fontsize=7.5, ncol=2,
              bbox_to_anchor=(0.5, -0.18), loc="upper center",
              framealpha=0.9, edgecolor="#e5e7eb")

    ax.set_xlabel("Inference latency — mean (ms) · lower is better →", fontsize=10)
    ax.set_ylabel("COCO val2017 mAP50-95 · higher is better", fontsize=10)
    ax.set_title("YOLOv8s Optimization Pareto: Latency vs Accuracy\n"
                 "RTX 4090 · COCO val2017 · batch=1", fontsize=11, pad=10)

    ax.set_xlim(0.2, 3.2)
    ax.set_ylim(0.422, 0.450)

    note = ("◆ Diamonds: fake-quant (weight-only INT8, FP32 inference latency).\n"
            "▲/■: compiled TRT engines. Error bars: mean → p99.")
    ax.text(0.99, 0.04, note, transform=ax.transAxes, fontsize=7,
            ha="right", va="bottom", color="#6b7280",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    out = FIGURES / "fig1_pareto.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 2: FP32 Overhead Breakdown + TRT comparison ───────────────────────

def fig2_profiling():
    fp32 = load("fp32_latency.json")
    fp16 = load("trt_fp16_latency.json")
    int8 = load("trt_int8_latency.json")

    # FP32 measured total = 2.453ms
    total_fp32 = fp32["timing"]["mean_ms"]

    # Breakdown derived from profile_nsys_stats.txt (documented in docs/profiling_analysis.md).
    # Kernel execution time = 30.5ms / 20 iters = 1.525ms
    # Dispatch gap = total - kernel_time = 0.928ms
    kernel_total_ms = 1.525
    dispatch_ms     = total_fp32 - kernel_total_ms   # 0.928ms

    # Kernel time fractions from nsys cuda_gpu_kern_sum (see docs/profiling_analysis.md)
    conv_ms   = kernel_total_ms * 0.580  # implicit GEMM + CUTLASS + Winograd
    bn_ms     = kernel_total_ms * 0.136
    layout_ms = kernel_total_ms * 0.104
    silu_ms   = kernel_total_ms * 0.077
    other_ms  = kernel_total_ms - conv_ms - bn_ms - layout_ms - silu_ms

    segments_fp32 = [
        ("Dispatch gap\n(CPU→GPU idle)",      dispatch_ms, C["dispatch"]),
        ("Conv compute\n(GEMM/Winograd)",     conv_ms,     C["conv"]),
        ("BatchNorm",                          bn_ms,       C["bn"]),
        ("Layout NCHW↔NHWC",                  layout_ms,   C["layout"]),
        ("SiLU activation",                    silu_ms,     C["silu"]),
        ("Other\n(pool, cat, upsample…)",      other_ms,    C["other"]),
    ]

    trt_fp16_ms = fp16["timing"]["mean_ms"]
    trt_int8_ms = int8["timing"]["mean_ms"]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    rows = [
        ("FP32 Eager\n(2.453 ms)", segments_fp32, total_fp32),
        ("TRT FP16\n(0.679 ms)",   None,          trt_fp16_ms),
        ("TRT INT8\n(0.533 ms)",   None,          trt_int8_ms),
    ]

    y_positions = [0.72, 0.42, 0.12]
    bar_height  = 0.18

    for (row_label, segs, total), y in zip(rows, y_positions):
        ax.text(-0.04, y, row_label, ha="right", va="center",
                fontsize=9, transform=ax.get_yaxis_transform())

        if segs is not None:
            # Full breakdown for FP32
            x = 0.0
            for seg_label, seg_ms, color in segs:
                ax.barh(y, seg_ms, height=bar_height, left=x,
                        color=color, edgecolor="white", linewidth=0.5)
                text_color = "#374151" if color == C["other"] else "white"
                if seg_ms > 0.20:
                    ax.text(x + seg_ms / 2, y, f"{seg_ms:.3f}ms",
                            ha="center", va="center", fontsize=7,
                            color=text_color, fontweight="bold")
                elif seg_ms > 0.05:  # narrow segments: rotate to fit within width
                    ax.text(x + seg_ms / 2, y, f"{seg_ms:.3f}ms",
                            ha="center", va="center", fontsize=6,
                            color=text_color, fontweight="bold", rotation=90)
                x += seg_ms
        else:
            # TRT bars: solid color, ms label inside, speedup label outside
            color = C["trt_fp16"] if "FP16" in row_label else C["trt_int8"]
            ax.barh(y, total, height=bar_height, color=color,
                    edgecolor="white", linewidth=0.5)
            ax.text(total / 2, y, f"{total:.3f} ms",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")
            speedup = total_fp32 / total
            ax.text(total + 0.02, y, f"{speedup:.2f}× faster",
                    va="center", fontsize=8.5, color=color, fontweight="bold")

    # Legend for FP32 segments
    legend_patches = [mpatches.Patch(color=s[2], label=s[0].replace("\n", " "))
                      for s in segments_fp32]
    ax.legend(handles=legend_patches, fontsize=7.5, loc="upper right",
              framealpha=0.9, edgecolor="#e5e7eb", ncol=2)

    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_xlim(0, total_fp32 * 1.45)
    ax.set_ylim(0, 1.0)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Where Does YOLOv8s FP32 Time Go?\n"
                 "FP32 breakdown from Nsight Systems · TRT via compiled engine + CUDA Graphs",
                 fontsize=11, pad=10)

    note = ("TRT FP16/INT8 eliminate the dispatch gap (CUDA Graphs),\n"
            "layout conversions (NHWC-native), and BN overhead (epilogue fusion).")
    ax.text(0.99, 0.03, note, transform=ax.transAxes, fontsize=7.5,
            ha="right", va="bottom", color="#6b7280",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    out = FIGURES / "fig2_profiling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 3: Custom Kernel Throughput + Roofline ────────────────────────────

def fig3_kernel():
    data = load("kernel_benchmark.json")
    shapes = data["per_shape_results"]
    peak_bw = data["peak_bw_gbs"]  # 1008 GB/s DRAM

    # Pair up PyTorch / CUDA kernel results
    labels, pt_bw, cu_bw, pt_us, cu_us, speedups = [], [], [], [], [], []
    pt_rows = [r for r in shapes if r["method"] == "PyTorch"]
    cu_rows = {r["label"]: r for r in shapes if r["method"] == "CUDA kernel"}

    short_labels = {
        "backbone_3x3_256":  "[256,256,3,3]",
        "neck_3x3_128":      "[128,128,3,3]",
        "small_3x3_64":      "[64,64,3,3]",
        "backbone_1x1_256":  "[256,256,1,1]",
        "neck_1x1_128x64":   "[128,64,1,1]",
        "dfl_conv":          "[1,16,1,1]\n(DFL)",
    }

    for r in pt_rows:
        lbl = r["label"]
        cu  = cu_rows[lbl]
        labels.append(short_labels[lbl])
        pt_bw.append(r["throughput_gbs"])
        cu_bw.append(cu["throughput_gbs"])
        pt_us.append(r["mean_us"])
        cu_us.append(cu["mean_us"])
        speedups.append(cu["speedup_vs_pytorch"])

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(bottom=0.22)

    ax.bar(x - width / 2, pt_bw, width, label="PyTorch baseline",
           color=C["pytorch"], edgecolor="white")
    ax.bar(x + width / 2, cu_bw, width,
           label="Custom CUDA kernel", color=C["cuda"], edgecolor="white")

    # Log scale — makes all six shapes readable at once
    ax.set_yscale("log")
    ax.set_ylim(0.003, 4000)

    # Roofline reference line
    ax.axhline(peak_bw, color="#ef4444", lw=1.4, ls="--", zorder=0)
    ax.text(len(labels) - 0.05, peak_bw * 1.08, f"DRAM peak ({peak_bw:.0f} GB/s)",
            ha="right", fontsize=8, color="#ef4444")

    # Speedup labels: above each CUDA bar; cap below DRAM line for bars that land close to it
    for i, (sp, cu_v) in enumerate(zip(speedups, cu_bw)):
        label = f"{sp:.1f}×\n(L2-cache-bound)" if i == 0 else f"{sp:.1f}×"
        label_y = cu_v * 1.6 if cu_v >= peak_bw else min(cu_v * 1.6, peak_bw * 0.62)
        ax.text(x[i] + width / 2, label_y, label,
                ha="center", va="bottom", fontsize=8.5,
                color=C["cuda"], fontweight="bold")

    # Launch-overhead annotation on the last two (tiny) shapes
    ax.text(x[-1], cu_bw[-1] * 5, "Launch-\nlatency\nbound",
            ha="center", va="bottom", fontsize=7.5, color="#6b7280", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Effective bandwidth — GB/s (log scale)", fontsize=10)
    ax.set_title("Per-Channel INT8 Fake-Quant: Custom CUDA Kernel vs PyTorch Baseline\n"
                 "RTX 4090 · 1000 timed reps · 50 warmup · bandwidth = (2× read + 1× write) / time",
                 fontsize=11, pad=10)
    ax.legend(fontsize=9, loc="lower left")

    note = ("★ [256,256,3,3] bar exceeds DRAM peak: 2.36 MB tensor fits in RTX 4090 L2 (72 MB) → 1624 GB/s is L2-BW-bound.\n"
            "★ [1,16,1,1] (DFL) is launch-latency-bound; 4.9× speedup comes entirely from eliminating 4 extra kernel dispatches.")
    ax.text(0.5, -0.17, note, transform=ax.transAxes, fontsize=7.5,
            ha="center", va="top", color="#6b7280",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9fafb", alpha=0.9))

    fig.tight_layout()
    out = FIGURES / "fig3_kernel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 4: Per-Layer INT8 Sensitivity ─────────────────────────────────────

def fig4_sensitivity():
    data     = load("quant_sensitivity.json")
    baseline = data["baseline_mAP50_95"]   # FP32 on 100-image subset
    results  = data["results"]             # one entry per conv layer

    # mAP_drop = baseline - mAP_when_only_this_layer_is_INT8
    # Positive = sensitive (quantizing this layer hurts accuracy)
    sorted_r = sorted(results, key=lambda r: r["mAP_drop"], reverse=True)

    layers   = [r["layer"] for r in sorted_r]
    drops    = [r["mAP_drop"] for r in sorted_r]
    n_params = [r["n_params"] for r in sorted_r]

    dfl_idx  = next(i for i, l in enumerate(layers) if "dfl" in l)

    # Color: DFL red, other sensitive layers orange, insensitive gray
    def bar_color(i, drop):
        if i == dfl_idx:
            return "#ef4444"    # red — DFL outlier
        if drop > 0.001:
            return "#f97316"    # orange — moderately sensitive
        return "#9ca3af"        # gray — insensitive (noise floor)

    colors = [bar_color(i, d) for i, d in enumerate(drops)]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(layers))
    bars = ax.bar(x, drops, color=colors, width=0.85, edgecolor="none")

    # Zero line
    ax.axhline(0, color="#374151", lw=0.8, zorder=3)

    # Noise floor band (±1 subset std, approximate)
    ax.axhspan(-0.002, 0.002, color="#f3f4f6", zorder=0, label="±0.002 subset noise floor")

    # Annotate the DFL bar (DFL is index 0 — leftmost; place text to the right)
    dfl_drop = drops[dfl_idx]
    dfl_params = n_params[dfl_idx]
    ax.annotate(
        f"model.22.dfl.conv\n{dfl_params} params — one scale\ncollapses learned CDF\n+{dfl_drop:.4f} mAP drop",
        xy=(dfl_idx, dfl_drop),
        xytext=(dfl_idx + 10, dfl_drop * 0.88),
        fontsize=8, color="#ef4444", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.2),
        ha="left",
    )

    # Second annotation: most expensive backbone layer (largest params, near-zero drop)
    expensive_idx = max(range(len(n_params)), key=lambda i: n_params[i])
    exp_layer_short = layers[expensive_idx].split(".")[-2] + "." + layers[expensive_idx].split(".")[-1]
    ax.annotate(
        f"Largest backbone conv\n({n_params[expensive_idx]//1024}K params)\nnear-zero sensitivity",
        xy=(expensive_idx, drops[expensive_idx]),
        xytext=(expensive_idx + 8, 0.005),
        fontsize=7.5, color="#6b7280",
        arrowprops=dict(arrowstyle="->", color="#6b7280", lw=0.8),
        ha="center",
    )

    # Legend patches
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color="#ef4444", label="DFL conv (model.22.dfl.conv) — most sensitive"),
        mpatches.Patch(color="#f97316", label="Moderately sensitive (drop > 0.001)"),
        mpatches.Patch(color="#9ca3af", label="Insensitive — safe to quantize"),
        mpatches.Patch(color="#f3f4f6", label="±0.002 subset noise floor"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
              framealpha=0.9, edgecolor="#e5e7eb")

    ax.set_xticks([])
    ax.set_xlabel(f"Each bar = one of {len(layers)} conv layers (sorted by sensitivity, high → low)",
                  fontsize=9)
    ax.set_ylabel("mAP drop when layer is quantized to INT8\n(all other layers remain FP32)", fontsize=9)
    ax.set_title("Per-Layer INT8 Sensitivity: Which Layers Actually Matter?\n"
                 "YOLOv8s · per-tensor symmetric weight-only INT8 · 100-image COCO subset",
                 fontsize=11, pad=10)
    ax.set_xlim(-1, len(layers))

    note = ("Key finding: 62/64 layers are insensitive to INT8 (drop < 0.002, within subset noise). "
            "The DFL conv is a 16-parameter outlier: one shared scale collapses\n"
            "its learned cumulative distribution function for bounding-box offsets. "
            "Protecting it (keeping FP32) recovers +0.0026 mAP on full val5000 — "
            "at near-zero compute cost.")
    ax.text(0.5, -0.20, note, transform=ax.transAxes, fontsize=7.5,
            ha="center", va="top", color="#374151",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9fafb", alpha=0.9))

    fig.subplots_adjust(bottom=0.28)
    out = FIGURES / "fig4_sensitivity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    FIGURES.mkdir(exist_ok=True)
    print("Generating figures...")
    try:
        fig1_pareto()
    except Exception as e:
        print(f"  fig1 FAILED: {e}", file=sys.stderr)
    try:
        fig2_profiling()
    except Exception as e:
        print(f"  fig2 FAILED: {e}", file=sys.stderr)
    try:
        fig3_kernel()
    except Exception as e:
        print(f"  fig3 FAILED: {e}", file=sys.stderr)
    try:
        fig4_sensitivity()
    except Exception as e:
        print(f"  fig4 FAILED: {e}", file=sys.stderr)
    print("Done.")


if __name__ == "__main__":
    main()
