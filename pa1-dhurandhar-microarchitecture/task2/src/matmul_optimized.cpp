// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"
// Horizontal sum of a __m256 (same reduction sequence as matmul_simd256).
static inline float hsum256_ps(__m256 v) {
    __m128 msb = _mm256_extractf128_ps(v, 1);
    __m128 lsb = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lsb, msb);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

void matmul_final(const float* A, const float* B, float* C,
                           int M, int N, int K, int lda, int ldb, int ldc) {
     // 1. Tiling parameters (Block sizes)
    const int TILE_M = 64;
    const int TILE_N = 64;
    const int TILE_K = 128;

    // 2. Prefetch parameters
    const int b_degree = 64;
    const int a_degree = 1;
    const int p_step = 16;              // must stay a multiple of 16 (2 x __m256)
    const int simd_width = 8;

    // Outer tile loops
    for (int i0 = 0; i0 < M; i0 += TILE_M) {
        int i_end = (i0 + TILE_M > M) ? M : i0 + TILE_M;
        for (int j0 = 0; j0 < N; j0 += TILE_N) {
            int j_end = (j0 + TILE_N > N) ? N : j0 + TILE_N;
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                int k_end = (k0 + TILE_K > K) ? K : k0 + TILE_K;
                // Inner element loops
                for (int i = i0; i < i_end; ++i) {
                    // Prefetch C's tile row exactly once per total K-accumulation
                    if (k0 == 0) {
                        _mm_prefetch(reinterpret_cast<const char*>(C + static_cast<long>(i) * ldc + j0), _MM_HINT_T0);
                    }
                    const float* a = A + static_cast<long>(i) * lda;

                    for (int j = j0; j < j_end; ++j) {
                        // If it's the first K-block, initialize to 0. Otherwise, accumulate previous K-blocks.
                        float acc = (k0 == 0) ? 0.0f : C[static_cast<long>(i) * ldc + j];
                        __m256 vacc0 = _mm256_setzero_ps();
                        __m256 vacc1 = _mm256_setzero_ps();
                        const float* b = B + static_cast<long>(j) * ldb;
                        int p = k0;
                        // Prefetch next row of A when we are reaching the end of the CURRENT j-tile.
                        // We also ensure i != i_end - 1 because if it's the last row of the tile,
                        // the next iteration jumps to a completely different tile block, making a+lda invalid.
                        bool prefetch_A_next_row = (j == j_end - a_degree) && (i != i_end - 1);

                        if (prefetch_A_next_row) {
                            for (p = k0; p <= k_end - p_step; p += p_step) {
                                _mm_prefetch(reinterpret_cast<const char*>(b + p + b_degree), _MM_HINT_T0);
                                _mm_prefetch(reinterpret_cast<const char*>(a + p + lda), _MM_HINT_T0);
                                const float* a_p = a + p;
                                const float* b_p = b + p;
                                vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_p), _mm256_loadu_ps(b_p), vacc0);
                                vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_p + simd_width),
                                                        _mm256_loadu_ps(b_p + simd_width), vacc1);
                            }
                        } else {
                            for (p = k0; p <= k_end - p_step; p += p_step) {
                                _mm_prefetch(reinterpret_cast<const char*>(b + p + b_degree), _MM_HINT_NTA);
                                const float* a_p = a + p;
                                const float* b_p = b + p;
                                vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_p), _mm256_loadu_ps(b_p), vacc0);
                                vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_p + simd_width),
                                                        _mm256_loadu_ps(b_p + simd_width), vacc1);
                            }
                        }
                        acc += hsum256_ps(_mm256_add_ps(vacc0, vacc1));

                        // Vector remainder: whole 8-wide chunks left over inside this K-block
                        for (; p <= k_end - simd_width; p += simd_width) {
                            acc += hsum256_ps(_mm256_mul_ps(_mm256_loadu_ps(a + p),
                                                            _mm256_loadu_ps(b + p)));
                        }
                        // Scalar remainder loop for edge-cases where TILE_K isn't divisible by simd_width
                        for (; p < k_end; p++) {
                            acc += a[p] * b[p];
                        }

                        // Store partial or final accumulation back to memory
                        C[static_cast<long>(i) * ldc + j] = acc;
                    }
                }
            }
        }
    }
}
void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    matmul_final(A, B, C, M, N, K, lda, ldb, ldc);
}
