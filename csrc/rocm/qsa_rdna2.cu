// T48 — QSA (Qwen Sparse Attention) decode glue for gfx1030.
//
// Three kernels for the Qwen4Exp QSA decode path. Two of them are ports
// of the corresponding Triton kernels in vllm/models/qwen4_exp/amd/ops/qsa.py;
// the third is a thin wrapper that re-uses the existing
// paged_mqa_logits_decode_rdna2 indexer kernel when the shapes line up.
//
//   qsa_store_cache_rows_rdna2
//       row-major rows -> paged K cache scatter (mirrors the 5D paged K
//       layout that reshape_and_cache_flash_rdna2 writes; reuses the same
//       stride contract).
//
//   qsa_compress_groups_rdna2
//       Average COMPRESS_RATIO consecutive keys into one pooled key. The
//       compression can span the per-request ring (compressor_state_cache)
//       and this step's raw_rows; a separate ring slot holds the
//       RoPE-position tail.
//
//   qsa_mqa_paged_rdna2
//       MQA paged logits: for each (row, kv_block), gather K rows, dot
//       with the row's Q (over NUM_HEADS index heads), and accumulate the
//       max(error=0) softmax-pre scores. The shape matches
//       paged_mqa_logits_decode_rdna2 (also: same per-row indexer layout),
//       so this host wrapper delegates to the existing HIP kernel when
//       the dtype/layout is compatible. Falls through to Triton otherwise
//       (the Python dispatcher in vllm/models/qwen4_exp/amd/ops/qsa.py
//       decides which path to take based on the env flag).
//
// Opt-in: VLLM_RDNA_QSA_HIP=1 + on_gfx10x() in the Python dispatcher.

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "rocm/ops.h"  // paged_mqa_logits_decode_rdna2

namespace {

__device__ __forceinline__ uint4 ld_u4(const void* p) {
  return *reinterpret_cast<const uint4*>(p);
}
__device__ __forceinline__ void st_u4(void* p, uint4 v) {
  *reinterpret_cast<uint4*>(p) = v;
}

// ---------------------------------------------------------------------------
// qsa_store_cache_rows_rdna2:
//   rows: [num_rows, WIDTH] fp16 (or bf16)
//   slots: [num_rows] int32 -> (block_id * PAGE_SIZE + token) in cache
//   cache: paged K cache, [num_blocks, PAGE_SIZE, WIDTH]
// Mirrors _store_qsa_rows_kernel in vllm/models/qwen4_exp/amd/ops/qsa.py.
// ---------------------------------------------------------------------------

template <int BLOCK_D>
__global__ void qsa_store_cache_rows_kernel(
    const char* __restrict__ rows, const int64_t* __restrict__ slots,
    char* __restrict__ cache, int num_rows, int num_blocks,
    int64_t stride_rows_row, int64_t stride_rows_dim,
    int64_t stride_cache_block, int64_t stride_cache_token,
    int64_t stride_cache_dim,
    int PAGE_SIZE, int WIDTH, int elem_bytes, int rows_per_block) {
  // The vectorized stride-1 path is only valid when both the row width and
  // the cache width are dense (element stride == 1). The server passes a
  // transposed ``canonical_qsa_rope_positions`` view whose last-dim stride is
  // ``tokens``, so fall back to the stride-aware scalar loop otherwise.
  const int elems_per_vec = 16 / elem_bytes;
  const bool row_dense = (stride_rows_dim == elem_bytes);
  const bool cache_dense = (stride_cache_dim == elem_bytes);
  const bool vec_exact = (WIDTH % elems_per_vec) == 0;
  if (row_dense && cache_dense && vec_exact && WIDTH >= elems_per_vec) {
    // Dense path: vec_count threads cover one row, so a block spans
    // blockDim.x / vec_count rows. One row per block left 240 of 256 threads
    // idle and cost ~2.5x the Triton kernel.
    const int vec_count = WIDTH / elems_per_vec;
    const int row = blockIdx.x * rows_per_block + threadIdx.x / vec_count;
    if (row >= num_rows) return;
    const int64_t slot = slots[row];
    if ((slot < 0) || (slot >= (int64_t)num_blocks * PAGE_SIZE)) return;
    char* cache_ptr = cache + (size_t)(slot / PAGE_SIZE) * stride_cache_block
        + (size_t)(slot % PAGE_SIZE) * stride_cache_token;
    const char* row_ptr = rows + (size_t)row * stride_rows_row;
    const int v = threadIdx.x % vec_count;
    st_u4(reinterpret_cast<uint4*>(cache_ptr + (size_t)v * 16),
          ld_u4(reinterpret_cast<const uint4*>(row_ptr + (size_t)v * 16)));
    return;
  }

  const int row = blockIdx.x;
  if (row >= num_rows) return;
  const int64_t slot = slots[row];
  if ((slot < 0) || (slot >= (int64_t)num_blocks * PAGE_SIZE)) return;
  char* cache_ptr = cache + (size_t)(slot / PAGE_SIZE) * stride_cache_block
      + (size_t)(slot % PAGE_SIZE) * stride_cache_token;
  const char* row_ptr = rows + (size_t)row * stride_rows_row;
  for (int i = threadIdx.x; i < WIDTH; i += blockDim.x) {
    const char* src = row_ptr + (size_t)i * stride_rows_dim;
    char* dst = cache_ptr + (size_t)i * stride_cache_dim;
    if (elem_bytes == 2) {
      *reinterpret_cast<half*>(dst) = *reinterpret_cast<const half*>(src);
    } else if (elem_bytes == 4) {
      *reinterpret_cast<int32_t*>(dst) = *reinterpret_cast<const int32_t*>(src);
    } else if (elem_bytes == 8) {
      *reinterpret_cast<int64_t*>(dst) = *reinterpret_cast<const int64_t*>(src);
    }
  }
}

void qsa_store_cache_rows(
    const at::Tensor& rows, const at::Tensor& slots, at::Tensor& cache,
    int64_t page_size, int64_t width) {
  TORCH_CHECK(rows.is_cuda() && slots.is_cuda() && cache.is_cuda(),
              "qsa_store_cache_rows_rdna2: all tensors on CUDA/HIP");
  TORCH_CHECK(rows.scalar_type() == cache.scalar_type(),
              "qsa_store_cache_rows_rdna2: rows and cache dtype must match");
  TORCH_CHECK(slots.scalar_type() == at::kLong,
              "qsa_store_cache_rows_rdna2: slots int64 only");
  TORCH_CHECK(rows.dim() == 2, "rows must be 2D");
  TORCH_CHECK(rows.size(1) == width, "rows.size(1) must equal width");
  const int num_rows = rows.size(0);
  const int num_blocks = cache.size(0);
  TORCH_CHECK(slots.size(0) == num_rows, "slots.size(0) must equal num_rows");

  const at::cuda::OptionalCUDAGuard guard(cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int elem_bytes = rows.element_size();
  TORCH_CHECK(cache.element_size() == elem_bytes,
              "rows and cache element sizes must match");
  TORCH_CHECK(elem_bytes == 2 || elem_bytes == 4 || elem_bytes == 8,
              "elem size must be 2, 4, or 8 bytes");
  const int64_t stride_rows_row = rows.stride(0) * elem_bytes;
  const int64_t stride_rows_dim = rows.stride(rows.dim() - 1) * elem_bytes;
  const int64_t stride_cache_block = cache.stride(0) * elem_bytes;
  const int64_t stride_cache_token = cache.stride(1) * elem_bytes;
  const int64_t stride_cache_dim = cache.stride(3) * elem_bytes;

  const int elems_per_vec = 16 / elem_bytes;
  const bool dense = (stride_rows_dim == elem_bytes) &&
      (stride_cache_dim == elem_bytes) && (width % elems_per_vec == 0) &&
      (width >= elems_per_vec);
  const int vec_count = dense ? (int)(width / elems_per_vec) : 1;
  const int rows_per_block = dense ? (256 / vec_count) : 1;
  const int grid = (num_rows + rows_per_block - 1) / rows_per_block;
  qsa_store_cache_rows_kernel<256><<<grid, 256, 0, stream>>>(
      reinterpret_cast<const char*>(rows.const_data_ptr()),
      slots.const_data_ptr<int64_t>(),
      reinterpret_cast<char*>(cache.mutable_data_ptr()),
      num_rows, num_blocks,
      stride_rows_row, stride_rows_dim,
      stride_cache_block, stride_cache_token, stride_cache_dim,
      (int)page_size, (int)width, elem_bytes, rows_per_block);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// qsa_compress_groups_rdna2:
//   Pooled[c, h] = (1/COMPRESS_RATIO) * sum_i K[group_i]
//   The group can span the per-request ring (compressor_state_cache) and
//   this step's raw_keys. We also write the first-group position tail.
// Mirrors _compress_qsa_groups_kernel in vllm/models/qwen4_exp/amd/ops/qsa.py.
// ---------------------------------------------------------------------------

template <int BLOCK_D, int COMPRESS_RATIO>
__global__ void qsa_compress_groups_kernel(
    const half* __restrict__ raw_keys, const int64_t* __restrict__ raw_positions,
    const half* __restrict__ compressor_state_cache,
    const int64_t* __restrict__ rope_cache,
    const int* __restrict__ compressor_state_table,
    const int* __restrict__ token_to_req,
    const int* __restrict__ query_start_loc,
    const int64_t* __restrict__ logical_positions,
    const int64_t* __restrict__ compressed_slots,
    half* __restrict__ pooled, int64_t* __restrict__ first_positions,
    int stride_raw_row, int stride_raw_dim,
    int stride_raw_positions_row, int stride_raw_positions_dim,
    int stride_compressor_state_block, int stride_compressor_state_token,
    int stride_compressor_state_dim,
    int stride_rope_block, int stride_rope_token, int stride_rope_dim,
    int stride_compressor_state_table_req,
    int stride_pooled_row, int stride_pooled_dim,
    int stride_positions_row, int stride_positions_dim,
    int num_rows, int num_compressor_state_blocks, int num_requests,
    int COMPRESSOR_STATE_SIZE, int HEAD_DIM,
    int LOAD_ROPE_POSITIONS, int DEBUG) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;

  const int request = token_to_req[row];
  const bool valid_request = (request >= 0) && (request < num_requests);
  const int safe_request = min(max(request, 0), num_requests - 1);
  const int query_row_start = valid_request
      ? query_start_loc[safe_request] : 0;
  const int query_row_end = valid_request
      ? query_start_loc[safe_request + 1] : 0;
  const int64_t end_position = logical_positions[row];
  const int64_t compressed_slot = compressed_slots[row];
  const int64_t chunk_start_position = end_position - (row - query_row_start);
  const int compressor_state_block =
      valid_request
          ? compressor_state_table[safe_request]
          : -1;
  const bool valid_compressor_state_block =
      (compressor_state_block >= 0)
      && (compressor_state_block < num_compressor_state_blocks);
  const bool valid_row =
      valid_request
      && (row >= query_row_start)
      && (row < query_row_end)
      && (end_position >= COMPRESS_RATIO - 1)
      && (compressed_slot >= 0);
  const int safe_compressor_block = max(compressor_state_block, 0);

  // One thread per output dim. A per-thread ``float acc[BLOCK_D]`` array spilled
  // to local memory, and every thread in the block recomputed the whole row.
  for (int b = threadIdx.x; b < HEAD_DIM; b += blockDim.x) {
    float acc = 0.f;
#pragma unroll
    for (int go = 0; go < COMPRESS_RATIO; go++) {
      const int64_t position = end_position - (COMPRESS_RATIO - 1 - go);
      const bool use_raw = position >= chunk_start_position;
      const int raw_row =
          (int)(query_row_start + position - chunk_start_position);
      const bool raw_ok = valid_row && use_raw
          && (raw_row >= query_row_start)
          && (raw_row < query_row_end)
          && (raw_row < num_rows);
      const bool state_ok = valid_row && !use_raw && valid_compressor_state_block;
      float v = 0.f;
      if (raw_ok) {
        v = __half2float(
            raw_keys[raw_row * stride_raw_row + b * stride_raw_dim]);
      } else if (state_ok) {
        const int tok = (int)(position % COMPRESSOR_STATE_SIZE);
        v = __half2float(
            compressor_state_cache[
                safe_compressor_block * stride_compressor_state_block
                + tok * stride_compressor_state_token
                + b * stride_compressor_state_dim]);
      }
      acc += v;
    }
    pooled[row * stride_pooled_row + b * stride_pooled_dim] =
        __float2half(acc / float(COMPRESS_RATIO));
  }

  // First-position tail: threads 0..2 own one dim each.
  const int64_t first_position = end_position - COMPRESS_RATIO + 1;
  if (threadIdx.x < 3) {
    const int p = (int)threadIdx.x;
    if (LOAD_ROPE_POSITIONS != 0) {
      const bool first_from_raw = first_position >= chunk_start_position;
      const int raw_first_row =
          (int)(query_row_start + first_position - chunk_start_position);
      const bool raw_first_ok = valid_row && first_from_raw
          && (raw_first_row >= query_row_start)
          && (raw_first_row < query_row_end)
          && (raw_first_row < num_rows);
      const bool state_first_ok = valid_row && !first_from_raw
          && valid_compressor_state_block;
      int64_t v = 0;
      if (raw_first_ok) {
        v = raw_positions[raw_first_row * stride_raw_positions_row
                          + p * stride_raw_positions_dim];
      } else if (state_first_ok) {
        const int tok = (int)(first_position % COMPRESSOR_STATE_SIZE);
        v = rope_cache[safe_compressor_block * stride_rope_block
                       + tok * stride_rope_token
                       + p * stride_rope_dim];
      }
      first_positions[row * stride_positions_row + p * stride_positions_dim] = v;
    } else {
      first_positions[row * stride_positions_row + p * stride_positions_dim] =
          valid_row ? first_position : (int64_t)0;
    }
  }
  __syncthreads();
  if (DEBUG && threadIdx.x == 0) {
    const int64_t v0 = first_positions[row * stride_positions_row + 0];
    const int64_t v1 = first_positions[row * stride_positions_row + 1];
    const int64_t v2 = first_positions[row * stride_positions_row + 2];
    const bool bad = (v0 < 0) || (v0 > 1000000) || (v1 < 0) || (v1 > 1000000)
        || (v2 < 0) || (v2 > 1000000);
    if (bad) {
      const int64_t r0 = rope_cache[safe_compressor_block * stride_rope_block
                                    + (int)(first_position % COMPRESSOR_STATE_SIZE)
                                        * stride_rope_token
                                    + 0 * stride_rope_dim];
      const int64_t r1 = rope_cache[safe_compressor_block * stride_rope_block
                                    + (int)(first_position % COMPRESSOR_STATE_SIZE)
                                        * stride_rope_token
                                    + 1 * stride_rope_dim];
      const int64_t r2 = rope_cache[safe_compressor_block * stride_rope_block
                                    + (int)(first_position % COMPRESSOR_STATE_SIZE)
                                        * stride_rope_token
                                    + 2 * stride_rope_dim];
      printf(
          "[QSA-KBAD] row=%d end=%lld chunk_start=%lld first_pos=%lld "
          "from_raw=%d raw_row=%d blk=%d tok=%d valid=%d "
          "v=(%lld,%lld,%lld) ring=(%lld,%lld,%lld) load_rope=%d\n",
          row, (long long)end_position, (long long)chunk_start_position,
          (long long)first_position,
          (int)(first_position >= chunk_start_position),
          (int)(query_row_start + first_position - chunk_start_position),
          safe_compressor_block, (int)(first_position % COMPRESSOR_STATE_SIZE),
          (int)valid_row, (long long)v0, (long long)v1, (long long)v2,
          (long long)r0, (long long)r1, (long long)r2, LOAD_ROPE_POSITIONS);
    }
  }
}

void qsa_compress_groups(
    const at::Tensor& raw_keys, const at::Tensor& raw_positions,
    const at::Tensor& compressor_state_cache, const at::Tensor& rope_cache,
    const at::Tensor& compressor_state_table,
    const at::Tensor& token_to_req, const at::Tensor& query_start_loc,
    const at::Tensor& logical_positions, const at::Tensor& compressed_slots,
    at::Tensor& pooled, at::Tensor& first_positions,
    int64_t compress_ratio, int64_t compressor_state_size,
    int64_t head_dim, bool load_rope_positions) {
  TORCH_CHECK(raw_keys.is_cuda(), "raw_keys must be CUDA");
  TORCH_CHECK(pooled.is_cuda(), "pooled must be CUDA");
  TORCH_CHECK(first_positions.is_cuda(), "first_positions must be CUDA");
  TORCH_CHECK(raw_keys.scalar_type() == at::kHalf, "raw_keys fp16 only");
  TORCH_CHECK(head_dim <= 128,
              "qsa_compress_groups_rdna2: head_dim <= 128 only (the kernel pads "
              "to BLOCK_D=128 and would silently drop wider rows); got ",
              head_dim);
  TORCH_CHECK(pooled.scalar_type() == at::kHalf, "pooled fp16 only");
  TORCH_CHECK(first_positions.scalar_type() == at::kLong,
              "first_positions int64 only");
  TORCH_CHECK(logical_positions.scalar_type() == at::kLong,
              "logical_positions int64 only");
  TORCH_CHECK(compressed_slots.scalar_type() == at::kLong,
              "compressed_slots int64 only");
  if (raw_positions.defined()) {
    TORCH_CHECK(raw_positions.scalar_type() == at::kLong,
                "raw_positions int64 only");
  }
  if (rope_cache.defined()) {
    TORCH_CHECK(rope_cache.scalar_type() == at::kLong,
                "rope_cache int64 only");
  }
  int64_t num_rows_i = raw_keys.size(0);
  num_rows_i = std::min<int64_t>(num_rows_i, logical_positions.size(0));
  num_rows_i = std::min<int64_t>(num_rows_i, token_to_req.size(0));
  num_rows_i = std::min<int64_t>(num_rows_i, compressed_slots.size(0));
  num_rows_i = std::min<int64_t>(num_rows_i, first_positions.size(0));
  num_rows_i = std::min<int64_t>(num_rows_i, pooled.size(0));
  if (raw_positions.defined()) {
    num_rows_i = std::min<int64_t>(num_rows_i, raw_positions.size(0));
  }
  if (num_rows_i != raw_keys.size(0)) {
    printf(
        "[QSA-WARN] row count mismatch: raw_keys=%lld clamped to %lld "
        "(logical_positions=%lld token_to_req=%lld compressed_slots=%lld "
        "first_positions=%lld pooled=%lld) — metadata buffers shorter than "
        "raw_keys\n",
        (long long)raw_keys.size(0), (long long)num_rows_i,
        (long long)logical_positions.size(0), (long long)token_to_req.size(0),
        (long long)compressed_slots.size(0), (long long)first_positions.size(0),
        (long long)pooled.size(0));
  }
  const int num_rows = (int)num_rows_i;
  const int num_compressor_state_blocks = compressor_state_cache.size(0);
  const int num_requests = compressor_state_table.size(0);
  const int BLOCK_D = 128;  // covers HEAD_DIM in {32, 64, 128}

  const at::cuda::OptionalCUDAGuard guard(raw_keys.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool qsa_dbg = ::getenv("VLLM_QSA_RDNA2_DEBUG") != nullptr;
  if (qsa_dbg) {
    printf(
        "[QSA-H] num_rows=%d cr=%lld cs=%lld hd=%lld stream=%p dev=%d "
        "fp=%p pooled=%p rk=%p\n",
        num_rows, (long long)compress_ratio, (long long)compressor_state_size,
        (long long)head_dim, (void*)stream,
        raw_keys.get_device(), (void*)first_positions.data_ptr(),
        (void*)pooled.data_ptr(), (void*)raw_keys.data_ptr());
  }
  auto launch = [&](auto cr_const) {
    constexpr int CR = decltype(cr_const)::value;
    qsa_compress_groups_kernel<BLOCK_D, CR><<<num_rows, 128, 0, stream>>>(
        reinterpret_cast<const half*>(raw_keys.const_data_ptr()),
        raw_positions.defined()
            ? raw_positions.const_data_ptr<int64_t>() : nullptr,
        compressor_state_cache.defined()
            ? reinterpret_cast<const half*>(compressor_state_cache.const_data_ptr())
            : nullptr,
        rope_cache.defined() ? rope_cache.const_data_ptr<int64_t>() : nullptr,
        compressor_state_table.const_data_ptr<int>(),
        token_to_req.const_data_ptr<int>(),
        query_start_loc.const_data_ptr<int>(),
        logical_positions.const_data_ptr<int64_t>(),
        compressed_slots.const_data_ptr<int64_t>(),
        reinterpret_cast<half*>(pooled.mutable_data_ptr()),
        first_positions.mutable_data_ptr<int64_t>(),
        (int)raw_keys.stride(0),
        (int)raw_keys.stride(2),
        raw_positions.defined() ? (int)raw_positions.stride(0) : 0,
        raw_positions.defined() ? (int)raw_positions.stride(2) : 0,
        (int)compressor_state_cache.stride(0),
        (int)compressor_state_cache.stride(1),
        (int)compressor_state_cache.stride(3),
        rope_cache.defined() ? (int)rope_cache.stride(0) : 0,
        rope_cache.defined() ? (int)rope_cache.stride(1) : 0,
        rope_cache.defined() ? (int)rope_cache.stride(3) : 0,
        (int)compressor_state_table.stride(0),
        (int)pooled.stride(0),
        (int)pooled.stride(2),
        (int)first_positions.stride(0),
        (int)first_positions.stride(1),
        num_rows, num_compressor_state_blocks, num_requests,
        (int)compressor_state_size, (int)head_dim,
        load_rope_positions ? 1 : 0, qsa_dbg ? 1 : 0);
  };
  if (compress_ratio == 1) { launch(std::integral_constant<int, 1>{}); }
  else if (compress_ratio == 2) { launch(std::integral_constant<int, 2>{}); }
  else if (compress_ratio == 4) { launch(std::integral_constant<int, 4>{}); }
  else if (compress_ratio == 8) { launch(std::integral_constant<int, 8>{}); }
  else if (compress_ratio == 16) { launch(std::integral_constant<int, 16>{}); }
  else { TORCH_CHECK(false, "qsa_compress_groups_rdna2: compress_ratio in {1,2,4,8,16} only"); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// ---------------------------------------------------------------------------
// Public host wrappers.
// ---------------------------------------------------------------------------

void qsa_store_cache_rows_rdna2(
    torch::Tensor rows, torch::Tensor slots, torch::Tensor cache,
    const at::Tensor& page_size, const at::Tensor& width) {
  qsa_store_cache_rows(rows, slots, cache,
                       page_size.item<int64_t>(), width.item<int64_t>());
}

void qsa_compress_groups_rdna2(
    torch::Tensor raw_keys, torch::Tensor raw_positions,
    torch::Tensor compressor_state_cache, torch::Tensor rope_cache,
    torch::Tensor compressor_state_table,
    torch::Tensor token_to_req, torch::Tensor query_start_loc,
    torch::Tensor logical_positions, torch::Tensor compressed_slots,
    torch::Tensor pooled, torch::Tensor first_positions,
    const at::Tensor& compress_ratio, const at::Tensor& compressor_state_size,
    const at::Tensor& head_dim, bool load_rope_positions) {
  qsa_compress_groups(
      raw_keys, raw_positions, compressor_state_cache, rope_cache,
      compressor_state_table, token_to_req, query_start_loc,
      logical_positions, compressed_slots,
      pooled, first_positions,
      compress_ratio.item<int64_t>(), compressor_state_size.item<int64_t>(),
      head_dim.item<int64_t>(), load_rope_positions);
}

// ---------------------------------------------------------------------------
// qsa_mqa_paged_rdna2: QSA indexer scoring, exact parity with the Triton
// `_qsa_mqa_paged_kernel` in vllm/models/qwen4_exp/amd/ops/qsa.py.
//
// The previous implementation delegated to `paged_mqa_logits_decode_rdna2`
// (the DeepSeek V4 indexer), whose contract is different in every respect:
// fp8 e4m3 q/K with per-slot dequant scales, H == 64 and D == 128 only,
// per-head weights, `block_tables` indexed by row, visibility from
// `context_lens`, and no `visible_blocks` output. QSA needs fp16, H = 4,
// D = 128, unit weights, a request-indirected page table, visibility derived
// from `compress_ratio`, and the `visible_blocks` side channel, so it gets a
// dedicated kernel here.
//
// One thread per output column, BLOCK_N columns per CTA.
// ---------------------------------------------------------------------------
namespace {

typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));

template <int BLOCK_N, typename TPos, typename TSeq>
__global__ void qsa_mqa_paged_kernel(
    const half* __restrict__ q, const half* __restrict__ k_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ token_to_req,
    const TPos* __restrict__ query_positions,
    const TSeq* __restrict__ sequence_lengths,
    int32_t* __restrict__ visible_blocks, float* __restrict__ logits,
    int64_t stride_q_row, int64_t stride_q_head, int64_t stride_q_dim,
    int64_t stride_cache_block, int64_t stride_cache_token,
    int64_t stride_cache_dim, int64_t stride_table_req,
    int64_t stride_table_page, int64_t stride_logits_row, int num_rows,
    int num_columns, int num_pages, int num_requests, int table_width,
    int page_size, int num_heads, int head_dim, int compress_ratio,
    float score_divisor) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;
  const int col = (int)blockIdx.y * BLOCK_N + (int)threadIdx.x;

  const int32_t request = token_to_req[row];
  const bool valid_request = (request >= 0) && (request < num_requests);
  const int safe_request = min(max(request, 0), num_requests - 1);
  const int64_t query_position = (int64_t)query_positions[row];
  const int64_t sequence_length =
      valid_request ? (int64_t)sequence_lengths[safe_request] : 0;
  const int64_t visible = min((query_position + 1) / compress_ratio,
                              sequence_length / compress_ratio);
  if (blockIdx.y == 0 && threadIdx.x == 0) {
    visible_blocks[row] = (int32_t)visible;
  }
  if (col >= num_columns) return;

  const int logical_page = col / page_size;
  const int page_offset = col % page_size;
  bool valid = valid_request && (col < visible) && (logical_page < table_width);
  int32_t physical_page = -1;
  if (valid) {
    const int safe_logical_page = min(logical_page, table_width - 1);
    physical_page = page_table[safe_request * stride_table_req
                               + safe_logical_page * stride_table_page];
    valid = (physical_page >= 0) && (physical_page < num_pages);
  }

  float score = 0.0f;
  if (valid) {
    const half* k_row = k_cache + (int64_t)physical_page * stride_cache_block
        + (int64_t)page_offset * stride_cache_token;
    const half* q_row = q + (int64_t)row * stride_q_row;
    // Dense contiguous rows: two MACs per instruction via v_dot2. The indexer
    // layout is [rows, heads, head_dim] contiguous with head_dim even, so this
    // is the production path; anything else falls back to scalar half loads.
    // The dense path takes 128 bits at a time and keeps four independent
    // fdot2 accumulator chains. The previous form issued one 4-byte load per
    // f16x2 operand (512 loads per column per row) and accumulated all 64
    // fdot2 of a head into a single register, so the dot was a 64-deep
    // dependent chain. That is why HIP lost to Triton once most columns were
    // visible: the early-chunk win came only from skipping masked columns, and
    // the dense case was simply a worse kernel.
    const bool dense = (stride_q_dim == 1) && (stride_cache_dim == 1)
        && (stride_q_head % 8 == 0) && (stride_cache_token % 8 == 0)
        && (head_dim % 8 == 0)
        && ((reinterpret_cast<uintptr_t>(k_row)
             | reinterpret_cast<uintptr_t>(q_row)) % 16 == 0);
    for (int h = 0; h < num_heads; ++h) {
      const half* q_head = q_row + (int64_t)h * stride_q_head;
      float dot = 0.0f;
      if (dense) {
        const uint4* q_vec = reinterpret_cast<const uint4*>(q_head);
        const uint4* k_vec = reinterpret_cast<const uint4*>(k_row);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int d8 = 0; d8 < head_dim / 8; ++d8) {
          const uint4 qq = q_vec[d8];
          const uint4 kk = k_vec[d8];
          const f16x2* qh = reinterpret_cast<const f16x2*>(&qq);
          const f16x2* kh = reinterpret_cast<const f16x2*>(&kk);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            acc[j] = __builtin_amdgcn_fdot2(qh[j], kh[j], acc[j], false);
          }
        }
        dot = (acc[0] + acc[1]) + (acc[2] + acc[3]);
      } else {
        for (int d = 0; d < head_dim; ++d) {
          dot += __half2float(q_head[(int64_t)d * stride_q_dim])
              * __half2float(k_row[(int64_t)d * stride_cache_dim]);
        }
      }
      score += fmaxf(dot, 0.0f);
    }
    score /= score_divisor;
  }
  const float neg_inf = __int_as_float(0xff800000);
  logits[(int64_t)row * stride_logits_row + col] = valid ? score : neg_inf;
}

template <typename TPos, typename TSeq>
void qsa_mqa_paged_launch(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& page_table,
    const at::Tensor& token_to_req, const at::Tensor& query_positions,
    const at::Tensor& sequence_lengths, at::Tensor& visible_blocks,
    at::Tensor& logits, int64_t compress_ratio, double score_divisor,
    cudaStream_t stream) {
  constexpr int kBlockN = 128;
  const int num_rows = q.size(0);
  const int num_columns = logits.size(1);
  const dim3 grid(num_rows, (num_columns + kBlockN - 1) / kBlockN);
  qsa_mqa_paged_kernel<kBlockN, TPos, TSeq><<<grid, kBlockN, 0, stream>>>(
      reinterpret_cast<const half*>(q.const_data_ptr()),
      reinterpret_cast<const half*>(k_cache.const_data_ptr()),
      page_table.const_data_ptr<int32_t>(),
      token_to_req.const_data_ptr<int32_t>(),
      query_positions.const_data_ptr<TPos>(),
      sequence_lengths.const_data_ptr<TSeq>(),
      visible_blocks.mutable_data_ptr<int32_t>(), logits.mutable_data_ptr<float>(),
      q.stride(0), q.stride(1), q.stride(2), k_cache.stride(0), k_cache.stride(1),
      k_cache.stride(3), page_table.stride(0), page_table.stride(1),
      logits.stride(0), num_rows, num_columns, k_cache.size(0),
      page_table.size(0), page_table.size(1), k_cache.size(1), q.size(1),
      q.size(2), (int)compress_ratio, (float)score_divisor);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace


std::tuple<at::Tensor, at::Tensor> qsa_mqa_paged_rdna2(
    const at::Tensor& q, const at::Tensor& k_cache,
    const at::Tensor& page_table, const at::Tensor& token_to_req,
    const at::Tensor& query_positions, const at::Tensor& sequence_lengths,
    int64_t compress_ratio, int64_t num_columns, double score_scale) {
  TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && page_table.is_cuda()
                  && token_to_req.is_cuda() && query_positions.is_cuda()
                  && sequence_lengths.is_cuda(),
              "qsa_mqa_paged_rdna2: all tensors on CUDA/HIP");
  TORCH_CHECK(q.scalar_type() == at::kHalf && k_cache.scalar_type() == at::kHalf,
              "qsa_mqa_paged_rdna2: fp16 q/k_cache only");
  TORCH_CHECK(q.dim() == 3, "qsa_mqa_paged_rdna2: q must be [rows, heads, dim]");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(2) == 1
                  && k_cache.size(3) == q.size(2),
              "qsa_mqa_paged_rdna2: k_cache must be [pages, page, 1, head_dim]");
  TORCH_CHECK(page_table.scalar_type() == at::kInt
                  && token_to_req.scalar_type() == at::kInt,
              "qsa_mqa_paged_rdna2: page_table/token_to_req int32 only");
  TORCH_CHECK(page_table.dim() == 2, "qsa_mqa_paged_rdna2: page_table must be 2D");
  TORCH_CHECK(token_to_req.numel() == q.size(0),
              "qsa_mqa_paged_rdna2: token_to_req must have one entry per row");
  TORCH_CHECK(query_positions.numel() == q.size(0),
              "qsa_mqa_paged_rdna2: query_positions must have one entry per row");
  TORCH_CHECK(sequence_lengths.numel() == page_table.size(0),
              "qsa_mqa_paged_rdna2: sequence_lengths must match page_table rows");
  TORCH_CHECK(compress_ratio > 0, "qsa_mqa_paged_rdna2: compress_ratio > 0");
  TORCH_CHECK(num_columns >= 0, "qsa_mqa_paged_rdna2: num_columns >= 0");
  TORCH_CHECK(score_scale > 0.0, "qsa_mqa_paged_rdna2: score_scale > 0");

  const auto pos_dtype = query_positions.scalar_type();
  const auto seq_dtype = sequence_lengths.scalar_type();
  TORCH_CHECK((pos_dtype == at::kLong || pos_dtype == at::kInt)
                  && (seq_dtype == at::kLong || seq_dtype == at::kInt),
              "qsa_mqa_paged_rdna2: query_positions/sequence_lengths must be int32 "
              "or int64");

  const int64_t rows = q.size(0);
  auto logits = at::empty({rows, num_columns},
                          at::dtype(at::kFloat).device(q.device()));
  auto visible_blocks = at::empty({rows}, at::dtype(at::kInt).device(q.device()));
  if (rows == 0 || num_columns == 0 || page_table.size(0) == 0) {
    return {logits, visible_blocks};
  }

  const at::cuda::OptionalCUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (pos_dtype == at::kLong && seq_dtype == at::kInt) {
    qsa_mqa_paged_launch<int64_t, int32_t>(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
        visible_blocks, logits, compress_ratio, score_scale, stream);
  } else if (pos_dtype == at::kLong) {
    qsa_mqa_paged_launch<int64_t, int64_t>(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
        visible_blocks, logits, compress_ratio, score_scale, stream);
  } else if (seq_dtype == at::kInt) {
    qsa_mqa_paged_launch<int32_t, int32_t>(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
        visible_blocks, logits, compress_ratio, score_scale, stream);
  } else {
    qsa_mqa_paged_launch<int32_t, int64_t>(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
        visible_blocks, logits, compress_ratio, score_scale, stream);
  }
  return {logits, visible_blocks};
}
