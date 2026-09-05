// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
// #include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;

            // 1. Initialize 4 independent 128-bit accumulators (16 floats total)
            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();
            __m128 acc2 = _mm_setzero_ps();
            __m128 acc3 = _mm_setzero_ps();

            int p = 0;

            // 2. Main loop: Unroll by 4x (16 floats per iteration)
            for (; p <= K - 16; p += 16) {
                // Load 16 elements from row A
                __m128 va0 = _mm_loadu_ps(a + p);
                __m128 va1 = _mm_loadu_ps(a + p + 4);
                __m128 va2 = _mm_loadu_ps(a + p + 8);
                __m128 va3 = _mm_loadu_ps(a + p + 12);

                // Load 16 elements from column B
                __m128 vb0 = _mm_loadu_ps(b + p);
                __m128 vb1 = _mm_loadu_ps(b + p + 4);
                __m128 vb2 = _mm_loadu_ps(b + p + 8);
                __m128 vb3 = _mm_loadu_ps(b + p + 12);

                // Multiply-accumulate independently across 4 hardware pipelines
                acc0 = _mm_add_ps(_mm_mul_ps(va0, vb0), acc0);
                acc1 = _mm_add_ps(_mm_mul_ps(va1, vb1), acc1);
                acc2 = _mm_add_ps(_mm_mul_ps(va2, vb2), acc2);
                acc3 = _mm_add_ps(_mm_mul_ps(va3, vb3), acc3);
            }

            // 3. Combine 4 accumulators into a single register
            __m128 acc = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));

            // 4. Clean vector loop for remaining 4-element blocks (K % 16 remainder)
            for (; p <= K - 4; p += 4) {
                __m128 va = _mm_loadu_ps(a + p);
                __m128 vb = _mm_loadu_ps(b + p);
                acc = _mm_add_ps(_mm_mul_ps(va, vb), acc);
            }

            // 5. Horizontal sum across the 4 elements of the combined acc register
            __m128 sum = _mm_hadd_ps(acc, acc);
            sum        = _mm_hadd_ps(sum, sum);
            float scalar_acc = _mm_cvtss_f32(sum);

            // 6. Scalar cleanup loop for remaining elements (K % 4 remainder)
            for (; p < K; ++p) {
                scalar_acc += a[p] * b[p];
            }

            C[static_cast<long>(i) * ldc + j] = scalar_acc;
        }
    }
}