// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
// #include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;

            // 1. Initialize a single 128-bit accumulator (4 float lanes)
            __m128 acc = _mm_setzero_ps();

            int p = 0;

            // 2. Main loop: Process 4 floats per iteration
            for (; p <= K - 4; p += 4) {
                __m128 va = _mm_loadu_ps(a + p);
                __m128 vb = _mm_loadu_ps(b + p);
                acc = _mm_add_ps(_mm_mul_ps(va, vb), acc);
            }

            // 3. Horizontal sum using _mm_hadd_ps
            __m128 sum = _mm_hadd_ps(acc, acc); // [x0+x1, x2+x3, x0+x1, x2+x3]
            sum        = _mm_hadd_ps(sum, sum); // [x0+x1+x2+x3, ...]
            
            float scalar_acc = _mm_cvtss_f32(sum);

            // 4. Scalar cleanup loop for remaining elements (when K % 4 != 0)
            for (; p < K; ++p) {
                scalar_acc += a[p] * b[p];
            }

            C[static_cast<long>(i) * ldc + j] = scalar_acc;
        }
    }
}