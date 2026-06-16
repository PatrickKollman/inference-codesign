/*
 * Per-channel INT8 fake-quantization CUDA kernel.
 *
 * Weight tensor shape: [C_out, spatial_size] (pre-flattened by caller).
 * One CUDA block per output channel. Two phases inside the block:
 *   Phase 1: parallel max-reduction over spatial_size elements → scale_c
 *   Phase 2: quantize each element using scale_c
 *
 * Memory passes: 2 (read for reduction, read+write for quant).
 * PyTorch naive baseline: ~5-6 passes (abs, max, div, round, clamp, mul).
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Warp-level max reduction using shuffle.
__device__ __forceinline__ float warp_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

/*
 * fake_quant_perchannel_kernel
 *
 * Grid:  (C_out,)
 * Block: (BLOCK_SIZE,) — must be a power of 2, <= 1024
 * Smem:  (BLOCK_SIZE / 32) * sizeof(float)  [one slot per warp]
 */
template <int BLOCK_SIZE>
__global__ void fake_quant_perchannel_kernel(
    const float* __restrict__ weight,
    float*       __restrict__ output,
    int spatial_size
) {
    const int c   = blockIdx.x;
    const int tid = threadIdx.x;

    const float* row     = weight + (long)c * spatial_size;
    float*       out_row = output + (long)c * spatial_size;

    // ── Phase 1: compute max(|row|) ──────────────────────────────────────
    float local_max = 0.0f;
    for (int i = tid; i < spatial_size; i += BLOCK_SIZE)
        local_max = fmaxf(local_max, fabsf(row[i]));

    // Warp reduction.
    local_max = warp_max(local_max);

    // One slot per warp in shared memory.
    constexpr int N_WARPS = BLOCK_SIZE / 32;
    __shared__ float smem[N_WARPS];
    if ((tid & 31) == 0)
        smem[tid >> 5] = local_max;
    __syncthreads();

    // Final reduction across warp leaders — N_WARPS <= 8, sequential is fine.
    if (tid == 0) {
        float v = smem[0];
        for (int i = 1; i < N_WARPS; i++)
            v = fmaxf(v, smem[i]);
        smem[0] = v;
    }
    __syncthreads();

    const float abs_max = smem[0];
    // Guard against all-zero channels (e.g. after pruning).
    const float scale   = (abs_max > 0.0f) ? (abs_max / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / scale;

    // ── Phase 2: quantize ─────────────────────────────────────────────────
    for (int i = tid; i < spatial_size; i += BLOCK_SIZE) {
        float q = rintf(row[i] * inv_scale);          // round-to-nearest-even
        q = fmaxf(-128.0f, fminf(127.0f, q));         // clamp to [-128, 127]
        out_row[i] = q * scale;
    }
}

/*
 * Dispatcher: choose BLOCK_SIZE based on spatial_size.
 * Rules:
 *   - Must be a power of 2.
 *   - >= 32 (at least one full warp, so N_WARPS >= 1 and smem is valid).
 *   - Capped at 256 (diminishing returns beyond that for BW-bound ops).
 */
static int choose_block_size(int spatial_size) {
    if (spatial_size <= 32)  return 32;
    if (spatial_size <= 64)  return 64;
    if (spatial_size <= 128) return 128;
    return 256;
}

/*
 * Python-callable entry point.
 *
 * Args:
 *   weight: float32 CUDA tensor, shape [C_out, spatial_size]
 * Returns:
 *   output: float32 CUDA tensor, same shape
 */
torch::Tensor fake_quant_perchannel_cuda(torch::Tensor weight) {
    TORCH_CHECK(weight.is_cuda(),               "weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat, "weight must be float32");
    TORCH_CHECK(weight.dim() == 2,              "weight must be 2D [C_out, spatial_size]");
    TORCH_CHECK(weight.is_contiguous(),         "weight must be contiguous");

    const int C_out        = weight.size(0);
    const int spatial_size = weight.size(1);

    auto output = torch::empty_like(weight);

    const float* w_ptr = weight.data_ptr<float>();
    float*       o_ptr = output.data_ptr<float>();

    const int bs = choose_block_size(spatial_size);
    // Shared memory: one float per warp.
    const int smem_bytes = (bs / 32) * sizeof(float);

    dim3 grid(C_out);

    switch (bs) {
        case  32: fake_quant_perchannel_kernel< 32><<<grid,  32, smem_bytes>>>(w_ptr, o_ptr, spatial_size); break;
        case  64: fake_quant_perchannel_kernel< 64><<<grid,  64, smem_bytes>>>(w_ptr, o_ptr, spatial_size); break;
        case 128: fake_quant_perchannel_kernel<128><<<grid, 128, smem_bytes>>>(w_ptr, o_ptr, spatial_size); break;
        default:  fake_quant_perchannel_kernel<256><<<grid, 256, smem_bytes>>>(w_ptr, o_ptr, spatial_size); break;
    }

    // Propagate any kernel errors immediately.
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fake_quant_perchannel kernel launch failed");

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fake_quant_perchannel",
        &fake_quant_perchannel_cuda,
        "Per-channel INT8 fake-quantize (CUDA). Input: float32 [C_out, spatial]. "
        "Returns float32 [C_out, spatial] with per-channel symmetric INT8 quant applied."
    );
}
