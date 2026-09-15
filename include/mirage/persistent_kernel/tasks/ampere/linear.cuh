/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "element_binary.cuh"
#include "element_unary.cuh"
#include "mma.cuh"
#include "reduction.cuh"
#include "smem_layout.cuh"
#include "tasks/common/common_header.cuh"

#define DEBUG 0

#if DEBUG
#define DCHECK(condition)                                                      \
  if ((condition) == 0) {                                                      \
    printf("Dcheck failed at %s:%d\n", __FILE__, __LINE__);                    \
  }
#else
#define DCHECK(condition)
#endif // DEBUG

namespace kernel {

using bfloat16 = type::bfloat16_t;
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE,
          int PIPE_MAX = 3,
          bool SAMPLE = false>
__device__ __forceinline__ void linear_kernel_impl(void const *input_ptr,
                                              void const *weight_ptr,
                                              void const *residual_ptr,
                                              void *output_ptr,
                                              int num_active_tokens,
                                              bool residual) {
  constexpr int CHUNK_SIZE = 16 / sizeof(T);
  constexpr int OUTPUT_ATOM_SIZE = OUTPUT_SIZE <= 64 ? OUTPUT_SIZE : 64;
  constexpr int log2_OUTPUT_ATOM_SIZE = log2_constexpr(OUTPUT_ATOM_SIZE);

  constexpr int TILE_SIZE = 128;
  constexpr int log2_TILE_SIZE = log2_constexpr(TILE_SIZE);
  constexpr int FORLOOP_RANGE = REDUCTION_SIZE / TILE_SIZE;

  constexpr int PIPE_FIT =
      PIPE_MAX < FORLOOP_RANGE ? PIPE_MAX : FORLOOP_RANGE;
  constexpr int ADJUSTED_PIPE_MAX = PIPE_FIT < 2 ? 2 : PIPE_FIT;

  constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_OUTPUT = BATCH_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_COL_B = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_ROW_C = OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int log2_CHUNK_SIZE = log2_constexpr(CHUNK_SIZE);
  constexpr int log2_CHUNKS_PER_ROW_A = log2_constexpr(CHUNKS_PER_ROW_A);
  constexpr int log2_CHUNKS_PER_COL_B = log2_constexpr(CHUNKS_PER_COL_B);
  constexpr int log2_CHUNKS_PER_ROW_C = log2_constexpr(CHUNKS_PER_ROW_C);

  // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
  // Each warp owns sixteen output columns, so a narrow atom cannot keep four
  // warps busy along N. The leftover warps go to K instead, which preserves
  // both correctness and the arithmetic intensity the narrow atom is for.
  constexpr int NUM_WARPS_N =
      OUTPUT_ATOM_SIZE < 64 ? (OUTPUT_ATOM_SIZE / 16) : 4;
  constexpr int NUM_WARPS_K = 4 / NUM_WARPS_N;
  static_assert(NUM_WARPS_N >= 1 && NUM_WARPS_N * NUM_WARPS_K == 4);
  static_assert(OUTPUT_ATOM_SIZE % 16 == 0);

  // TODO: support NUM_ITERS_M > 1, i.e., BATCH_SIZE > 16
  constexpr int NUM_ITERS_M = 1;
  constexpr int NUM_ITERS_N =
      (OUTPUT_SIZE + OUTPUT_ATOM_SIZE - 1) / OUTPUT_ATOM_SIZE;
  constexpr int NUM_ITERS_K =
      (TILE_SIZE + NUM_WARPS_K * 16 - 1) / (NUM_WARPS_K * 16);
  // constexpr int NUM_ITERS_K = 8;

  constexpr int log2_NUM_WARPS_N = log2_constexpr(NUM_WARPS_N);
  constexpr int log2_NUM_ITERS_K = log2_constexpr(NUM_ITERS_K);

  int warp_idx = warp_id();
  int warp_row = warp_idx >> log2_NUM_WARPS_N;
  int warp_col = warp_idx & (NUM_WARPS_N - 1);
  int lane_idx = lane_id();

  // The caller's count is launch-wide; this tile only has BATCH_SIZE rows.
  num_active_tokens =
      num_active_tokens < BATCH_SIZE ? num_active_tokens : BATCH_SIZE;

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T const *__restrict__ d_residual = static_cast<T const *>(residual_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);
  // CANNOT perform residual when redisual_ptr is nullptr
  if (residual_ptr == nullptr) {
    assert(!residual);
  }

  using InputDmem = dmem_row_const<T, BATCH_SIZE, TILE_SIZE, REDUCTION_SIZE>;
  using WeightDmem =
      dmem_col_const<T, TILE_SIZE, OUTPUT_ATOM_SIZE, REDUCTION_SIZE>;
  using ResidualDmem = dmem_row_const<T, BATCH_SIZE, OUTPUT_SIZE, O_STRIDE>;
  using OutputDmem = dmem_row<T, BATCH_SIZE, OUTPUT_SIZE, O_STRIDE>;

  InputDmem input_dmem(d_input);
  WeightDmem weight_dmem(d_weight);
  ResidualDmem residual_dmem(d_residual);
  OutputDmem output_dmem(d_output);

  extern __shared__ char smem_cta[];
  char *smem = mpk_smem(smem_cta);

  // STensors' offsets
  constexpr size_t ZERO_BUFFER_OFFSET = 0;
  // sizeof(T) * 8

  constexpr size_t SHARED_INPUT_BUFFER_OFFSET =
      ZERO_BUFFER_OFFSET + sizeof(T) * 64;
  // sizeof(T) * BATCH_SIZE * TILE_SIZE

  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET =
      SHARED_INPUT_BUFFER_OFFSET +
      sizeof(T) * BATCH_SIZE * ADJUSTED_PIPE_MAX * TILE_SIZE;

  constexpr size_t SHARED_OUTPUT_OFFSET =
      // MM_INTERMEDIATE_OFFSET +
      SHARED_WEIGHT_BUFFER_OFFSET +
      sizeof(T) * TILE_SIZE * ADJUSTED_PIPE_MAX * OUTPUT_ATOM_SIZE;

  // zero buffer
  T *zero_buf = (T *)(smem + ZERO_BUFFER_OFFSET);
  vec_zero_t<T, 8>::fill_zero(zero_buf);

  // copy
  T *shared_input_buffer = (T *)(smem + SHARED_INPUT_BUFFER_OFFSET);
  T *shared_weight_buffer = (T *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);

  // output
  T *shared_output = (T *)(smem + SHARED_OUTPUT_OFFSET);

  // define the swizzle mode
  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using InputSmem =
      smem_row_2dcol<T, 3, 3, 3, BATCH_SIZE, TILE_SIZE, ADJUSTED_PIPE_MAX>;
  using WeightSmem = smem_col_2drow<T,
                                    3,
                                    3,
                                    3,
                                    TILE_SIZE,
                                    OUTPUT_ATOM_SIZE,
                                    ADJUSTED_PIPE_MAX>;
  using OutputFullSmem =
      smem_row<T, 3, 3, 3, BATCH_SIZE, OUTPUT_ATOM_SIZE, OUTPUT_ATOM_SIZE>;

  // we no longger need zero buffer, but we could keep it to make sure shared
  // memory was aligned.
  ZeroBufferSmem zero_buffer(zero_buf);

  InputSmem input_smem(shared_input_buffer);
  WeightSmem weight_smem(shared_weight_buffer);

  OutputFullSmem output_smem(shared_output);

#pragma unroll
  for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
    // If we use NUM_ITERS_M and NUM_ITERS_N inside NUM_ITERS_K, the
    // loop for NUM_ITERS_K couldn't be unrolled in nvcc which hurts
    // performance.
#pragma unroll
    for (uint32_t nn = 0; nn < NUM_ITERS_N; nn++) {
      float s_frag[8];

      // should we sync here? if NUM_ITERS_N > 1, I suppose we should do it,
      // because we will write output_smem later, but it may be still used in
      // some warp which are still write to gmem.
      if (NUM_ITERS_N > 1) {
        mpk_group_sync();
      }
      // Initialize output_smem: if residual is provided, preload it; otherwise
      // zero
#pragma unroll
      for (int i = mpk_tid(); i < BATCH_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
           i += NUM_THREADS) {
        int row = i / (OUTPUT_ATOM_SIZE / CHUNK_SIZE);
        int dst_col = (i % (OUTPUT_ATOM_SIZE / CHUNK_SIZE)) << log2_CHUNK_SIZE;
        int src_col = dst_col + (nn << log2_OUTPUT_ATOM_SIZE);
        // TODO: use ignore-src in load_smem to avoid if-else
        if (residual) {
          load_smem(output_smem(row, dst_col), residual_dmem(row, src_col));
        } else {
          *((__uint128_t *)((void *)&output_smem.at(row, dst_col))) = 0ul;
        }
      }

      // initialize registers
#pragma unroll
      for (uint32_t r = 0; r < 8; r++) {
        s_frag[r] = 0;
      }

      int ismem_read_stage = 0;
      int ismem_write_stage = 0;

      // Warm up weight and input tiles for the first ADJUSTED_PIPE_MAX - 1
      // tile.
#pragma unroll
      for (int istage = 0; istage < ADJUSTED_PIPE_MAX - 1; ++istage) {
        // we don't need module for ADJUSTED_PIPE_MAX here, because we just load
        // ADJUSTED_PIPE_MAX - 1 pipe.
        int src_stage_offset = istage << log2_TILE_SIZE;

#pragma unroll
        for (int chunk = 0; chunk < NUM_CHUNKS_A / NUM_THREADS; chunk++) {
          int tid = mpk_tid();
          int threadCol = (tid & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
          int threadRow = tid >> log2_CHUNKS_PER_ROW_A;
          constexpr int ROWS_PER_ITERATION = NUM_THREADS / CHUNKS_PER_ROW_A;

          int dst_col = threadCol;
          int src_col = dst_col + src_stage_offset;

          int row_within = threadRow + chunk * ROWS_PER_ITERATION;
          int src_row = row_within;
          int dst_row = row_within;

          load_smem(input_smem(dst_row, dst_col, istage),
                    input_dmem(src_row, src_col));
        }
#pragma unroll
        for (int chunk = 0; chunk < NUM_CHUNKS_B / NUM_THREADS; chunk++) {
          int tid = mpk_tid();
          int threadRow = (tid & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
          int threadCol = tid >> log2_CHUNKS_PER_COL_B;
          constexpr int COLS_PER_ITERATION = NUM_THREADS / CHUNKS_PER_COL_B;

          int dst_row = threadRow;
          int src_row = dst_row + src_stage_offset;

          int col_within = threadCol + chunk * COLS_PER_ITERATION;
          int src_col = (nn << log2_OUTPUT_ATOM_SIZE) + col_within;
          int dst_col = col_within;

          load_smem(weight_smem(dst_row, dst_col, istage),
                    weight_dmem(src_row, src_col));
        }
        cp_async_fence();

        ++ismem_write_stage;
      } // warm up for ADJUSTED_PIPE_MAX - 1

      constexpr int PIPE_INSIDE_TILE = 2;
      uint32_t a_frag[PIPE_INSIDE_TILE][4], b_frag[PIPE_INSIDE_TILE][4];
      // wait for first warm up pipeline cp.async finished
      cp_async_wait<ADJUSTED_PIPE_MAX - 2>();
      mpk_group_sync();

      int warmup_m_col =
          (warp_row << (4 + log2_NUM_ITERS_K)) + ((lane_idx >> 4) << 3);
      int warmup_n_row =
          (warp_row << (4 + log2_NUM_ITERS_K)) + (((lane_idx & 0xF) >> 3) << 3);
      int warmup_smem_row = (lane_idx & 0xF);
      int warmup_n_col =
          (warp_col << 4) + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
      T *warmup_input_ptr = input_smem(warmup_smem_row, warmup_m_col, 0);
      DCHECK(warmup_n_col < OUTPUT_ATOM_SIZE);
      T *warmup_weight_ptr = weight_smem(warmup_n_row, warmup_n_col, 0);

      ldsm(warmup_input_ptr, a_frag[0]);
      ldsm(warmup_weight_ptr, b_frag[0]);

      // One lane times the K loop: the interval between tiles is II, and what
      // cp.async could not hide shows up as the wait below.
      bool const _samp = SAMPLE && (warp_idx == 0) && (lane_idx == 0) &&
                         (nn == 0);
      unsigned long long _prev = 0, _tw = 0, _ii = 0, _stall = 0;
      int _nsamp = 0;
#pragma unroll 1
      for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
        if (_samp && for_idx >= MPK_PROFILE_SKIP &&
            for_idx < FORLOOP_RANGE - 2) {
          unsigned long long _t = clock64();
          if (_prev != 0) {
            _ii += _t - _prev;
            ++_nsamp;
          }
          _prev = _t;
        }
#pragma unroll
        for (int k = 0; k < NUM_ITERS_K; k++) {
          // TODO(Wenqin): use pointer advance for the pointer for input and
          // weight shared memory instead of address calculation for
          // input_smem and weight_smem, because in each iteration in the K
          // dim for the OUTER_ROW/COL, they just advanced a compile-time know
          // offset, and it seems the CUTLASS version just use some ADD inst to
          // do it.
          int k_next = (k + 1) % NUM_ITERS_K;

          if (k == 0) {
            // loading next tile (for_idx + ADJUSTED_PIPE_MAX - 1) when k is 0.
            if (for_idx + ADJUSTED_PIPE_MAX - 1 < FORLOOP_RANGE) {
              int src_stage_offset = (for_idx + ADJUSTED_PIPE_MAX - 1)
                                     << log2_TILE_SIZE;
              // Prefetch next weight tile into ring buffer stage_write
              // Load input tile at the first output tile
#pragma unroll
              for (int chunk = 0; chunk < NUM_CHUNKS_A / NUM_THREADS; chunk++) {
                // we don't need to hoist the threadCol and threadRow,,
                // accorrding to experiment, the nvcc could hoist these const.
                int tid = mpk_tid();
                int threadCol = (tid & (CHUNKS_PER_ROW_A - 1))
                                << log2_CHUNK_SIZE;
                int threadRow = tid >> log2_CHUNKS_PER_ROW_A;
                constexpr int ROWS_PER_ITERATION =
                    NUM_THREADS / CHUNKS_PER_ROW_A; // 8

                int dst_col = threadCol;
                int src_col = dst_col + src_stage_offset;

                int row_within = threadRow + chunk * ROWS_PER_ITERATION;
                int src_row = row_within;
                int dst_row = row_within;

                load_smem(input_smem(dst_row, dst_col, ismem_write_stage),
                          input_dmem(src_row, src_col));
              }
#pragma unroll
              for (int chunk = 0; chunk < NUM_CHUNKS_B / NUM_THREADS; chunk++) {
                int tid = mpk_tid();
                int threadRow = (tid & (CHUNKS_PER_COL_B - 1))
                                << log2_CHUNK_SIZE;
                int threadCol = tid >> log2_CHUNKS_PER_COL_B;
                constexpr int COLS_PER_ITERATION =
                    NUM_THREADS / CHUNKS_PER_COL_B; // 8

                int dst_row = threadRow;
                int src_row = dst_row + src_stage_offset;

                int col_within = threadCol + chunk * COLS_PER_ITERATION;
                int src_col = (nn << log2_OUTPUT_ATOM_SIZE) + col_within;
                int dst_col = col_within;

                load_smem(weight_smem(dst_row, dst_col, ismem_write_stage),
                          weight_dmem(src_row, src_col));
              }
              ismem_write_stage = (ismem_write_stage + 1) % ADJUSTED_PIPE_MAX;
            }
            cp_async_fence();
          } // k == 0 for load next tile

          if (k == NUM_ITERS_K - 1) {
            // wait cp.async because we will load next tile data in to regs
            // when k == NUM_ITERS_K - 1.
            bool const _sw = _samp && for_idx >= MPK_PROFILE_SKIP &&
                             for_idx < FORLOOP_RANGE - 2 && _nsamp > 0;
            if (_sw) {
              _tw = clock64();
            }
            if (FORLOOP_RANGE - for_idx > 2) {
              cp_async_wait<ADJUSTED_PIPE_MAX - 2>();
            } else {
              cp_async_wait<0>();
            }
            mpk_group_sync();
            // Closed after the barrier on purpose: cp_async_wait is per
            // thread and lane 0 waits only on what it committed, but the group
            // cannot move until its slowest member has waited too.
            if (_sw) {
              _stall += clock64() - _tw;
            }

            // TODO(Wenqin): The comment out code below here is what we could
            // do for just use ADD for input and weight shared memory pointer.
            // int tmp_ismem_read_stage = ismem_read_stage;
            ismem_read_stage = (ismem_read_stage + 1) % ADJUSTED_PIPE_MAX;
            // input_ptr += (ismem_read_stage - tmp_ismem_read_stage) * (8 *
            // 128); weight_ptr += (ismem_read_stage - tmp_ismem_read_stage) *
            // (64 * 128);
          } // k == NUM_ITERS_K - 1

          static_assert(NUM_ITERS_M == 1);

          int m_row = (lane_idx & 0xF) + (m << 4);
          int n_col =
              (warp_col << 4) + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
          DCHECK(n_col < OUTPUT_ATOM_SIZE);

          int m_col = (warp_row << (4 + log2_NUM_ITERS_K)) + (k_next << 4) +
                      ((lane_idx >> 4) << 3);
          int n_row = (warp_row << (4 + log2_NUM_ITERS_K)) + (k_next << 4) +
                      (((lane_idx & 0xF) >> 3) << 3);

          int smem_row = m_row;
          T *valid_input_ptr = input_smem(smem_row, m_col, ismem_read_stage);
          // we don't need to check for is_input_valid, because we will use
          // num_active_tokens for the output, we will just pick valid output.
          T *input_ptr = valid_input_ptr;

          T *valid_weight_ptr = weight_smem(n_row, n_col, ismem_read_stage);
          T *weight_ptr = valid_weight_ptr;

          ldsm(input_ptr, a_frag[(k + 1) % PIPE_INSIDE_TILE]);
          ldsm(weight_ptr, b_frag[(k + 1) % PIPE_INSIDE_TILE]);
          mma_m16n16k16_bf16bf16bf32(s_frag,
                                     a_frag[k % PIPE_INSIDE_TILE],
                                     b_frag[k % PIPE_INSIDE_TILE],
                                     s_frag);

        } // loop for NUM_ITERS_K
      }   // loop for FORLOOP_RANGE

      if (_samp && _nsamp > 0) {
        constexpr int _SID = MPK_SHAPE_ID(log2_constexpr(BATCH_SIZE),
                                          log2_constexpr(OUTPUT_ATOM_SIZE),
                                          log2_constexpr(FORLOOP_RANGE));
        static_assert(_SID >= 0 && _SID < MPK_NUM_SHAPES,
                      "shape id out of range");
        mpk_a_report(_SID, ADJUSTED_PIPE_MAX, _ii / _nsamp, _stall / _nsamp);
      }

      // Warps that share a warp_col hold partial sums over disjoint K
      // ranges and land on identical (m, n) fragments, so one warp_row
      // accumulates at a time and the loop doubles as the K reduction.
#pragma unroll
      for (int wk = 0; wk < NUM_WARPS_K; wk++) {
        if (NUM_WARPS_K > 1) {
          mpk_group_sync();
        }
        if (warp_row == wk) {
#pragma unroll
          for (uint32_t i = 0; i < 4; i++) {
            int row_in_warp = (lane_idx >> 2) + ((i & 0x1) << 3);
            int col_within =
                (warp_col << 4) + ((lane_idx & 0x3) << 1) + ((i >> 1) << 3);
            int col = col_within;
            DCHECK(col_within < OUTPUT_ATOM_SIZE);
            if (row_in_warp < num_active_tokens) {
              // TODO: try st.matrix here?
              output_smem.at(row_in_warp, col) += bfloat16(s_frag[(i << 1)]);
              output_smem.at(row_in_warp, col + 1) +=
                  bfloat16(s_frag[(i << 1) | 0x1]);
            }
          }
        }
      }
      mpk_group_sync();

      // Final writeback: store accumulated output (residual already included if
      // any)
#pragma unroll
      for (int i = mpk_tid(); i < NUM_CHUNKS_OUTPUT; i += NUM_THREADS) {
        int row = i / CHUNKS_PER_ROW_C;
        int src_col = (i % CHUNKS_PER_ROW_C) << log2_CHUNK_SIZE;
        int dst_col = src_col + (nn << log2_OUTPUT_ATOM_SIZE);
        *((__uint128_t *)((void *)&output_dmem.at(row, dst_col))) =
            *((__uint128_t *)((void *)&output_smem.at(row, src_col)));
      }
    } // loop for NUM_ITERS_N, it may not be 1
  }   // loop for NUM_ITERS_M, it should always be 1, no sense loop
}


// Exact arena footprint of the impl below, mirroring its offset arithmetic.
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE,
          int PIPE>
struct LinearFootprint {
  static constexpr int ATOM = OUTPUT_SIZE <= 64 ? OUTPUT_SIZE : 64;
  static constexpr int FL = REDUCTION_SIZE / 128;
  static constexpr int P_FIT = PIPE < FL ? PIPE : FL;
  static constexpr int P = P_FIT < 2 ? 2 : P_FIT;
  static constexpr int bytes =
      (int)(sizeof(T) * 64 + sizeof(T) * BATCH_SIZE * P * 128 +
            sizeof(T) * 128 * P * ATOM + sizeof(T) * BATCH_SIZE * ATOM);
};

// All three depths are compiled in; which one runs depends on what the arena
// can still hold when this task is dispatched.
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE,
          int PIPE_MAX = 3>
__device__ __forceinline__ void linear_kernel(void const *input_ptr,
                                              void const *weight_ptr,
                                              void const *residual_ptr,
                                              void *output_ptr,
                                              int num_active_tokens,
                                              bool residual) {
  extern __shared__ char smem_cta[];
  using F0 = LinearFootprint<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE,
                             MPK_DEEP_PIPE>;
  using F1 = LinearFootprint<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE,
                             MPK_MID_PIPE>;
  using F2 = LinearFootprint<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE,
                             MPK_SHALLOW_PIPE>;
  constexpr bool HEAVY = (REDUCTION_SIZE / 128) >= MPK_HEAVY_FORLOOP;
#if MPK_SEL_POLICY == 5
  // Decided from this instantiation's own template parameters, so the claim
  // below sees one constant candidate and, at one group per CTA, folds away
  // entirely.
  constexpr int _PREF =
      mpk_static_pick(mpk_a_static(F2::ATOM, F2::FL), F2::FL, F0::bytes,
                      F1::bytes, F2::bytes, F0::P, F1::P, F2::P,
                      MPK_ARENA_USABLE);
  int const need[3] = {_PREF == 0 ? F0::bytes : 0, _PREF == 1 ? F1::bytes : 0,
                       _PREF == 2 ? F2::bytes : 0};
#elif MPK_SEL_POLICY >= 3
  // The profile decides, so every compiled depth stays on the menu.
  int const need[3] = {F0::bytes, F1::bytes, F2::bytes};
#else
  // Heavy tasks offer only the deep variant so they end up alone; light ones
  // offer the packable depths and share.
  int const need[3] = {HEAVY ? F0::bytes : 0, HEAVY ? 0 : F1::bytes,
                       HEAVY ? 0 : F2::bytes};
#endif
  if (mpk_tid() == 0) {
    int pref = -1;
#if MPK_SEL_POLICY == 5
    pref = _PREF;
#elif MPK_SEL_POLICY >= 3
    // Depths come from the footprints, not the macros, so a variant clamped by
    // a short reduction loop is scored at the depth it will really run.
    int const depth[3] = {F0::P, F1::P, F2::P};
    constexpr int _SID = MPK_SHAPE_ID(log2_constexpr(BATCH_SIZE),
                                      log2_constexpr(F2::ATOM),
                                      log2_constexpr(F2::FL));
    static_assert(_SID >= 0 && _SID < MPK_NUM_SHAPES,
                  "shape id out of range");
    // Probing runs the shallowest variant deliberately. Sampling at the depth
    // the score already likes can only confirm an upper bound: a covered
    // pipeline has no stall to measure, so a comes back at zero and the score
    // concludes that depth was enough. The shallowest variant is the one most
    // likely to run short of loads, which is what makes the stall observable.
    // Hoisted out of the guarded region below: the sampling bit is read after
    // it, whichever arm compiles.
    bool _probe = false;
#if MPK_SEL_DEBUG == 1
    pref = 2;
#elif MPK_SEL_DEBUG == 2
    pref = -1;
#else
    // What this CTA already decided for this shape. A hit ends the matter
    // without touching global memory, which is the whole point: the two loads
    // it replaces sit on the one thread the entire group waits behind, and at
    // ~550ns per task they cost several times what the depth choice wins.
    signed char *const _pc = mpk_pick_arr(smem_cta);
#if MPK_REMAIN_W
    // Read off the cell the worker loop wrote, on this same thread.
    int const _rb = mpk_rem_bucket(mpk_rem_arr(smem_cta)[mpk_group_id()]);
#else
    int const _rb = 0;
#endif
    int const _key = _SID * MPK_PICK_BUCKETS + (_rb % MPK_PICK_BUCKETS);
    int const _cached =
        (MPK_SCORE_LIVE || MPK_TENANCY_W) ? 0 : _pc[_key];
    if (_cached > 0) {
      pref = _cached - 1;
    } else {
      bool _final = false;
      _probe = mpk_should_sample(_SID, &_final);
      if (!_probe) {
        // An unmeasured shape reads as a = 0 rather than declining to score.
        // That is the honest prior -- assume the loads are covered until a
        // stall says otherwise -- and it is the only answer available for a
        // reduction too short to sample, since the window [SKIP, FL-2) is
        // empty once FL drops to 4. It is also the right answer there: with no
        // stall to hide, the prologue term decides and the shallowest live
        // depth wins.
        // Seeded here rather than on every task: it is a property of the
        // shape, the score below is its only reader, and an unconditional
        // atomic on one shared address is what the other 143 workers are
        // queueing behind.
        {
          int const _sf = need[2] > 0 ? need[2] : need[1];
          atomicMax(&mpk_co_foot, _sf);
#if MPK_CO_FOOT_MIN
          atomicMin(&mpk_co_foot_lo, _sf);
#endif
        }
        int const _a = mpk_a_ema[_SID];
#if MPK_REMAIN_W
        int const _w = mpk_bucket_w(_rb);
#else
        int const _w = mpk_crit_w(_SID);
#endif
        pref = mpk_score_pick(smem_cta, _a > 0 ? _a - 1 : 0, F2::FL, need,
                              depth, _w);
        mpk_k_sel[_SID] = 1 + pref;
#if MPK_RW_HIST
        mpk_rw_note(mpk_rem_arr(smem_cta)[mpk_group_id()], pref);
#endif
        // Only once the budget is spent, so no CTA freezes a decision taken
        // from an estimate still moving under it.
        // Exact, not an approximation: with the weight quantised to the
        // bucket the answer is a function of the key and nothing else.
        if (_final && !MPK_SCORE_LIVE && !MPK_TENANCY_W) {
          _pc[_key] = (signed char)(1 + pref);
        }
      } else {
        // Shallowest candidate the report will accept. Below depth 3 the wait
        // drains the pipeline on every tile, so the stall carries no
        // information about coverage and mpk_a_report drops it.
        pref = (depth[2] >= 3) ? 2 : ((depth[1] >= 3) ? 1 : 0);
      }
    }
#endif // MPK_SEL_DEBUG
#endif // MPK_SEL_POLICY >= 3
    mpk_wsel_note(mpk_wsel_req, pref);
    mpk_claim(smem_cta, need, pref);
#if MPK_SEL_POLICY >= 3 && MPK_SEL_POLICY != 5
    // Rides in bit 2 of the selection cell so the group picks it up from the
    // barrier below instead of needing another broadcast.
    if (_probe) {
      mpk_sel_arr(smem_cta)[mpk_group_id()] |= 4;
    }
#endif
  }
  mpk_group_sync();
  int const _cell = mpk_selected(smem_cta);
  int const sel = _cell & 3;
  if (mpk_tid() == 0) {
    mpk_wsel_note(mpk_wsel_got, sel);
  }
#if MPK_SEL_POLICY == 5
  // Nothing samples, so the instrumented instantiation is unreachable and the
  // dispatch below compiles to one call per depth instead of two.
  constexpr bool smp = false;
#else
  bool const smp = (_cell & 4) != 0;
#endif
#define MPK_LIN_RUN(PIPE)                                                      \
  do {                                                                         \
    if (smp) {                                                                 \
      linear_kernel_impl<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, \
                         PIPE, true>(input_ptr, weight_ptr, residual_ptr,      \
                                     output_ptr, num_active_tokens, residual); \
    } else {                                                                   \
      linear_kernel_impl<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, \
                         PIPE, false>(input_ptr, weight_ptr, residual_ptr,     \
                                      output_ptr, num_active_tokens,           \
                                      residual);                               \
    }                                                                          \
  } while (0)
  if (sel == 0) {
    MPK_LIN_RUN(MPK_DEEP_PIPE);
  } else if (sel == 1) {
    MPK_LIN_RUN(MPK_MID_PIPE);
  } else {
    MPK_LIN_RUN(MPK_SHALLOW_PIPE);
  }
#undef MPK_LIN_RUN
  mpk_group_sync();
  if (mpk_tid() == 0) {
    mpk_release(smem_cta);
  }
}
} // namespace kernel
