// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"
#include <cstdio>
#include "utils.h"

void conv_opt_scalar_range(const float* in, float* out, const float* ker, int oy,
                           int xa, int xb, int W, int in_stride, int K) {
    for (int ox = xa; ox < xb; ++ox) {
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

void conv_optimized_v2(const float* in, float* out, const float* ker,
                       int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int BW = 32;
    const int W_blocked = (W / BW) * BW;
    const int H_blocked = (H / 2) * 2;   // two output rows per iteration

    for (int oy = 0; oy < H_blocked; oy += 2) {
        for (int ox = 0; ox < W_blocked; ox += BW) {
            __m256 a0 = _mm256_setzero_ps();   // top output row
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();
            __m256 b0 = _mm256_setzero_ps();   // bottom output row
            __m256 b1 = _mm256_setzero_ps();
            __m256 b2 = _mm256_setzero_ps();
            __m256 b3 = _mm256_setzero_ps();

            // r walks the K+1 input rows the 2-row block needs.
            for (int r = 0; r < K + 1; ++r) {
                const int in_offset = (oy + r) * in_stride + ox;
                const bool useA = (r < K);    // feeds the top row,    kernel row r
                const bool useB = (r >= 1);   // feeds the bottom row, kernel row r-1

                for (int kx = 0; kx < K; ++kx) {
                    // one load, up to two FMAs -- this is the whole point of v2
                    const __m256 v0 = _mm256_loadu_ps(&in[in_offset + kx]);
                    const __m256 v1 = _mm256_loadu_ps(&in[in_offset + kx + 8]);
                    const __m256 v2 = _mm256_loadu_ps(&in[in_offset + kx + 16]);
                    const __m256 v3 = _mm256_loadu_ps(&in[in_offset + kx + 24]);

                    if (useA) {
                        const __m256 ka = _mm256_set1_ps(ker[r * K + kx]);
                        a0 = _mm256_fmadd_ps(ka, v0, a0);
                        a1 = _mm256_fmadd_ps(ka, v1, a1);
                        a2 = _mm256_fmadd_ps(ka, v2, a2);
                        a3 = _mm256_fmadd_ps(ka, v3, a3);
                    }
                    if (useB) {
                        const __m256 kb = _mm256_set1_ps(ker[(r - 1) * K + kx]);
                        b0 = _mm256_fmadd_ps(kb, v0, b0);
                        b1 = _mm256_fmadd_ps(kb, v1, b1);
                        b2 = _mm256_fmadd_ps(kb, v2, b2);
                        b3 = _mm256_fmadd_ps(kb, v3, b3);
                    }
                }
            }

            float* o0 = &out[oy * W + ox];
            float* o1 = o0 + W;
            _mm256_storeu_ps(o0 + 0, a0);
            _mm256_storeu_ps(o0 + 8, a1);
            _mm256_storeu_ps(o0 + 16, a2);
            _mm256_storeu_ps(o0 + 24, a3);
            _mm256_storeu_ps(o1 + 0, b0);
            _mm256_storeu_ps(o1 + 8, b1);
            _mm256_storeu_ps(o1 + 16, b2);
            _mm256_storeu_ps(o1 + 24, b3);
        }
        // leftover columns for both rows of the block
        conv_opt_scalar_range(in, out, ker, oy, W_blocked, W, W, in_stride, K);
        conv_opt_scalar_range(in, out, ker, oy + 1, W_blocked, W, W, in_stride, K);
    }
    // leftover output row when H is odd
    for (int oy = H_blocked; oy < H; ++oy) {
        conv_opt_scalar_range(in, out, ker, oy, 0, W, W, in_stride, K);
    }
}

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
    conv_optimized_v2(in, out, ker, H, W, K);
}

#ifdef STANDALONE_TEST
#include <cstdlib>
#include <cstring>
 
// Standalone profiling driver.
//
//   usage: <prog> [H [W [K [seed [variant]]]]]
//
// Every argument is optional and each is guarded independently, so a short
// argv can never make us read past the end of the array (the earlier
// "if (argc >= 4)" guard read argv[4], and conv_tile read argv[5], both of
// which are out of bounds when exactly four arguments are supplied).
//
// The final checksum print keeps the optimiser from deleting the convolution
// call: `out` is otherwise never read, so at -O2 the whole kernel is dead code.
 
int main(int argc, char** argv) {
    int H = 2048, W = 2048, K = 3;
    unsigned seed = 1234;
    const char* variant = "default";
 
    if (argc >= 2) H = std::atoi(argv[1]);
    if (argc >= 3) W = std::atoi(argv[2]);
    if (argc >= 4) K = std::atoi(argv[3]);
    if (argc >= 5) seed = static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10));
    if (argc >= 6) variant = argv[5];
 
    std::printf("PROFILING %s variant=%s H=%d W=%d K=%d seed=%u\n",
                "optimized", variant, H, W, K, seed);
 
    float* img = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    float* ker = pa1::alloc_floats(static_cast<std::size_t>(K) * K);
    float* out = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
 
    pa1::fill_random(img, static_cast<std::size_t>(H) * W, seed);
    pa1::fill_random(ker, static_cast<std::size_t>(K) * K, seed + 1u);
    float* in = pa1::make_padded(img, H, W, K);  // zero-padded halo buffer, stride W+2p
 
    if (std::strcmp(variant, "default") != 0 && std::strcmp(variant, "base") != 0) {
        std::fprintf(stderr, "unknown variant '%s' (this file has a single implementation)\n", variant);
        return 2;
    }
    conv_optimized(in, out, ker, H, W, K);
 
    std::printf("checksum %.6f\n",
                static_cast<double>(out[0])
                    + static_cast<double>(out[static_cast<std::size_t>(H) * W - 1])
                    + static_cast<double>(in[0]));
    return 0;
}
#endif
