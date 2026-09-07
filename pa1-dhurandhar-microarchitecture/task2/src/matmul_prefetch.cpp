// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch_v1(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc){
    const int b_degree=128;//Always a multiple of p_step as this is over the row of B itself
    const int a_degree=1;//Atleast 1.In terms of rows of B. How many rows of B are remaining when I should fetch next row of A?
    const int p_step=16;
    for (int i = 0; i < M; ++i) {
        _mm_prefetch(reinterpret_cast<const char*>(C + static_cast<long>(i) * ldc), _MM_HINT_T0);
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;

            int p=0;
            if (j == N - a_degree && i != M - 1) {
                for (p = 0; p <= K - p_step; p += p_step) {
                    _mm_prefetch(reinterpret_cast<const char*>(b + p + b_degree), _MM_HINT_NTA);
                    _mm_prefetch(reinterpret_cast<const char*>(a + p + lda), _MM_HINT_T0);
                    const float* a_p=a+p;
                    const float* b_p=b+p;                  
                    for(int iter=0; iter<p_step; iter++) acc += a_p[iter] * b_p[iter];
                }
            } else {
                for (p = 0; p <= K - p_step; p += p_step) {
                    _mm_prefetch(reinterpret_cast<const char*>(b + p + b_degree), _MM_HINT_NTA);
                    const float* a_p=a+p;
                    const float* b_p=b+p;
                    for(int iter=0; iter<p_step; iter++) acc += a_p[iter] * b_p[iter];
                }
            }
            for(;p<K;p++){
                acc+=a[p] * b[p];
            }

            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}

void matmul_prefetch_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K, int lda, int ldb, int ldc) {
    // 1. Tiling parameters (Block sizes)
    // Tweak these based on L1/L2 cache sizes. 
    // 32x32x128 elements * 4 bytes = 16KB per matrix tile (fits beautifully in L1d)
    const int TILE_M = 32;
    const int TILE_N = 32;
    const int TILE_K = 32; 

    // 2. Prefetch parameters
    const int b_degree = 64;
    const int a_degree = 1;
    const int p_step = 16;

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
                        
                        const float* b = B + static_cast<long>(j) * ldb;
                        int p = k0;
                        
                        // Prefetch next row of A when we are reaching the end of the CURRENT j-tile.
                        // We also ensure i != i_end - 1 because if it's the last row of the tile, 
                        // the next iteration jumps to a completely different tile block, making a+lda invalid.
                        bool prefetch_A_next_row = (j == j_end - a_degree) && (i != i_end - 1);

                        if (prefetch_A_next_row) {
                            for (p = k0; p <= k_end - p_step; p += p_step) {
                                _mm_prefetch(reinterpret_cast<const char*>(b + p + b_degree), _MM_HINT_NTA);
                                _mm_prefetch(reinterpret_cast<const char*>(a + p + lda), _MM_HINT_T0);
                                
                                const float* a_p = a + p;
                                const float* b_p = b + p;                  
                                for(int iter = 0; iter < p_step; iter++) acc += a_p[iter] * b_p[iter];
                            }
                        } else {
                            for (p = k0; p <= k_end - p_step; p += p_step) {
                                _mm_prefetch(reinterpret_cast<const char*>(b + p + b_degree), _MM_HINT_NTA);
                                
                                const float* a_p = a + p;
                                const float* b_p = b + p;
                                for(int iter = 0; iter < p_step; iter++) acc += a_p[iter] * b_p[iter];
                            }
                        }
                        
                        // Remainder loop for edge-cases where TILE_K isn't perfectly divisible by p_step
                        for(; p < k_end; p++){
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

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    matmul_prefetch_tiled(A, B, C, M, N, K, lda, ldb, ldc);
}
