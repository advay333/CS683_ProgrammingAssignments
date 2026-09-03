// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"
void matmul_simd128(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    const int simd_width = 4;
    const int tail = K % simd_width;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m128 acc = _mm_setzero_ps();
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K - tail; p+= simd_width) {
                __m128 va = _mm_loadu_ps(&a[p]);
                __m128 vb = _mm_loadu_ps(&b[p]);
                acc = _mm_fmadd_ps(va, vb, acc);
            }
            acc = _mm_hadd_ps(acc, acc);
            acc = _mm_hadd_ps(acc, acc);
            float sum = _mm_cvtss_f32(acc);    
            for (int p = K - tail; p < K; ++p) {
                sum += a[p] * b[p];
            }
            C[static_cast<long>(i) * ldc + j] = sum;
        }
    }
}
void matmul_simd256(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    const int simd_width = 8;
    const int tail = K % simd_width;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256 acc = _mm256_setzero_ps();
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K - tail; p+= simd_width) {
                __m256 va = _mm256_loadu_ps(&a[p]);
                __m256 vb = _mm256_loadu_ps(&b[p]);
                acc = _mm256_fmadd_ps(va, vb, acc);
            }
            __m128 msb = _mm256_extractf128_ps(acc, 1);
            __m128 lsb = _mm256_castps256_ps128(acc);
            __m128 sum128 = _mm_add_ps(lsb, msb);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);

            
            float sum = _mm_cvtss_f32(sum128);
            for (int p = K - tail; p < K; ++p) {
                sum += a[p] * b[p];
            }
            C[static_cast<long>(i) * ldc + j] = sum;
        }
    }
}
void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    matmul_simd128(A, B, C, M, N, K, lda, ldb, ldc);
}
