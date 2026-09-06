// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch_v1(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc){
    const int b_degree=128;//Always a multiple of 16 as this is over the row of B itself
    const int a_degree=1;//Atleast 1.In terms of rows of B. How many rows of B are remaining when I should fetch next row of A?
    for (int i = 0; i < M; ++i) {
        _mm_prefetch(reinterpret_cast<const char*>(C + static_cast<long>(i) * ldc), _MM_HINT_T0);
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;

            int p=0;
            if (j == N - a_degree && i != M - 1) {
                for (p = 0; p <= K - 32; p += 32) {
                    _mm_prefetch(reinterpret_cast<const char*>(b + p + b_degree), _MM_HINT_NTA);
                    _mm_prefetch(reinterpret_cast<const char*>(a + p + lda), _MM_HINT_T0);
                    const float* a_p=a+p;
                    const float* b_p=b+p;                  
                    for(int iter=0; iter<32; iter++) acc += a_p[iter] * b_p[iter];
                }
            } else {
                for (p = 0; p <= K - 32; p += 32) {
                    _mm_prefetch(reinterpret_cast<const char*>(b + p + b_degree), _MM_HINT_NTA);
                    const float* a_p=a+p;
                    const float* b_p=b+p;
                    for(int iter=0; iter<32; iter++) acc += a_p[iter] * b_p[iter];
                }
            }
            for(;p<K;p++){
                acc+=a[p] * b[p];
            }

            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    matmul_prefetch_v1(A, B, C, M, N, K, lda, ldb, ldc);
}
