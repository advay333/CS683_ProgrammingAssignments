// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"



void conv_simd128_v1(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int tail = W % 4;
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox <= W - 4; ox+=4) {
            __m128 acc = _mm_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m128 in_val = _mm_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    __m128 ker_val = _mm_set1_ps(ker[ky * K + kx]);
                    acc = _mm_fmadd_ps(ker_val, in_val, acc);
                }
            }
            _mm_storeu_ps(&out[oy * W + ox], acc);
        }
        for(int ox = W - tail; ox < W; ++ox){
            float acc  = 0.0f;
            for(int ky = 0; ky < K; ++ky) {
                const int in_offset = (oy + ky) * in_stride + ox;
                const int ker_offset = ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    acc += ker[ker_offset + kx] * in[in_offset + kx];
                }
            }
        }
    }
}
void conv_simd128_v2(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int tail = W % 4;
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox <= W - 4; ox+=4) {
            __m128 acc = _mm_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const int in_offset = (oy + ky) * in_stride + ox;
                const int ker_offset = ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    __m128 in_val = _mm_loadu_ps(&in[in_offset + kx]);
                    __m128 ker_val = _mm_set1_ps(ker[ker_offset + kx]);
                    acc = _mm_fmadd_ps(ker_val, in_val, acc);
                }
            }
            _mm_storeu_ps(&out[oy * W + ox], acc);
        }
        for(int ox = W - tail; ox < W; ++ox){
            float acc  = 0.0f;
            for(int ky = 0; ky < K; ++ky) {
                const int in_offset = (oy + ky) * in_stride + ox;
                const int ker_offset = ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    acc += ker[ker_offset + kx] * in[in_offset + kx];
                }
            }
        }
    }
}
void conv_simd128_v3(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int tail = W % 8;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox <= W - 8; ox+=8) {
            __m128 acc1 = _mm_setzero_ps();
            __m128 acc2 = _mm_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const int in_offset = (oy + ky) * in_stride + ox;
                const int ker_offset = ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    __m128 in_val1 = _mm_loadu_ps(&in[in_offset + kx]);
                    __m128 in_val2 = _mm_loadu_ps(&in[in_offset + kx + 4]);
                    __m128 ker_val = _mm_set1_ps(ker[ker_offset + kx]);
                    acc1 = _mm_fmadd_ps(ker_val, in_val1, acc1);
                    acc2 = _mm_fmadd_ps(ker_val, in_val2, acc2);
                }
            }
            _mm_storeu_ps(&out[oy * W + ox], acc1);
            _mm_storeu_ps(&out[oy * W + ox + 4], acc2);
        }
        for(int ox = W - tail; ox < W; ++ox){
            float acc  = 0.0f;
            for(int ky = 0; ky < K; ++ky) {
                const int in_offset = (oy + ky) * in_stride + ox;
                const int ker_offset = ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    acc += ker[ker_offset + kx] * in[in_offset + kx];
                }
            }
        }
    }
}
void conv_simd256_v1(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int tail = W % 8;
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox <= W - 8; ox+=8) {
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const int in_offset = (oy + ky) * in_stride + ox;
                const int ker_offset = ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    __m256 in_val = _mm256_loadu_ps(&in[in_offset + kx]);
                    __m256 ker_val = _mm256_set1_ps(ker[ker_offset + kx]);
                    acc = _mm256_fmadd_ps(ker_val, in_val, acc);
                }
            }
            _mm256_storeu_ps(&out[oy * W + ox], acc);
        }
        for(int ox = W - tail; ox < W; ++ox){
            float acc  = 0.0f;
            for(int ky = 0; ky < K; ++ky) {
                const int in_offset = (oy + ky) * in_stride + ox;
                const int ker_offset = ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    acc += ker[ker_offset + kx] * in[in_offset + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}

void conv_simd256_v2(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int par = 6;
    const int tail = W % (8*par);
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox <= W - 8*par; ox+=8*par) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const int in_offset = (oy + ky) * in_stride + ox;
                const int ker_offset = ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    __m256 ker_val = _mm256_set1_ps(ker[ker_offset + kx]);
                    __m256 in_val0 = _mm256_loadu_ps(&in[in_offset + kx]);
                    __m256 in_val1 = _mm256_loadu_ps(&in[in_offset + kx + 8]);
                    __m256 in_val2 = _mm256_loadu_ps(&in[in_offset + kx + 16]);
                    __m256 in_val3 = _mm256_loadu_ps(&in[in_offset + kx + 24]);
                    __m256 in_val4 = _mm256_loadu_ps(&in[in_offset + kx + 32]);
                    __m256 in_val5 = _mm256_loadu_ps(&in[in_offset + kx + 40]);
                    acc0 = _mm256_fmadd_ps(ker_val, in_val0, acc0);
                    acc1 = _mm256_fmadd_ps(ker_val, in_val1, acc1);
                    acc2 = _mm256_fmadd_ps(ker_val, in_val2, acc2);
                    acc3 = _mm256_fmadd_ps(ker_val, in_val3, acc3);
                    acc4 = _mm256_fmadd_ps(ker_val, in_val4, acc4);
                    acc5 = _mm256_fmadd_ps(ker_val, in_val5, acc5);
                }
            }
            _mm256_storeu_ps(&out[oy * W + ox], acc0);
            _mm256_storeu_ps(&out[oy * W + ox + 8], acc1);
            _mm256_storeu_ps(&out[oy * W + ox + 16], acc2);
            _mm256_storeu_ps(&out[oy * W + ox + 24], acc3);
            _mm256_storeu_ps(&out[oy * W + ox + 32], acc4);
            _mm256_storeu_ps(&out[oy * W + ox + 40], acc5);
        }
        for(int ox = W - tail; ox < W; ++ox){
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const int in_offset = (oy + ky) * in_stride + ox;
                const int ker_offset = ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    acc += ker[ker_offset + kx] * in[in_offset + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}
void conv_simd256_v3(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int par = 2;
    const int tail = W % (8*par);
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox <= W - 8*par; ox+=8*par) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            // __m256 acc2 = _mm256_setzero_ps();
            // __m256 acc3 = _mm256_setzero_ps();
            // __m256 acc4 = _mm256_setzero_ps();
            // __m256 acc5 = _mm256_setzero_ps();
            // __m256 acc6 = _mm256_setzero_ps();
            // __m256 acc7 = _mm256_setzero_ps();
            // __m256 acc8 = _mm256_setzero_ps();
            // __m256 acc9 = _mm256_setzero_ps();
            // __m256 acc10 = _mm256_setzero_ps();
            // __m256 acc11 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const int in_offset = (oy + ky) * in_stride + ox;
                const int ker_offset = ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    __m256 ker_val = _mm256_set1_ps(ker[ker_offset + kx]);

                    acc0 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx]), acc0);
                    acc1 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 8]), acc1);
                    // acc2 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 16]), acc2);
                    // acc3 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 24]), acc3);
                    // acc4 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 32]), acc4);
                    // acc5 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 40]), acc5);
                    // acc6 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 48]), acc6);
                    // acc7 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 56]), acc7);
                    // acc8 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 64]), acc8);
                    // acc9 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 72]), acc9);
                    // acc10 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 80]), acc10);
                    // acc11 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 88]), acc11);
                }
            }
            _mm256_storeu_ps(&out[oy * W + ox], acc0);
            _mm256_storeu_ps(&out[oy * W + ox + 8], acc1);
            // _mm256_storeu_ps(&out[oy * W + ox + 16], acc2);
            // _mm256_storeu_ps(&out[oy * W + ox + 24], acc3);
            // _mm256_storeu_ps(&out[oy * W + ox + 32], acc4);
            // _mm256_storeu_ps(&out[oy * W + ox + 40], acc5);
            // _mm256_storeu_ps(&out[oy * W + ox + 48], acc6);
            // _mm256_storeu_ps(&out[oy * W + ox + 56], acc7);
            // _mm256_storeu_ps(&out[oy * W + ox + 64], acc8);
            // _mm256_storeu_ps(&out[oy * W + ox + 72], acc9);
            // _mm256_storeu_ps(&out[oy * W + ox + 80], acc10);
            // _mm256_storeu_ps(&out[oy * W + ox + 88], acc11);
        }
        for(int ox = W - tail; ox < W; ++ox){
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const int in_offset = (oy + ky) * in_stride + ox;
                const int ker_offset = ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    acc += ker[ker_offset + kx] * in[in_offset + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}
void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    conv_simd256_v3(in, out, ker, H, W, K);
    // conv_naive(in, out, ker, H, W, K);
}
