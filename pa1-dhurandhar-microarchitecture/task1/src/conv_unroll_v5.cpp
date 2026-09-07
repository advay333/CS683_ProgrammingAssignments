// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"
#include <cstdio> // For profiling purposes explained below
#include "utils.h" // For profiling purposes explained below



void conv_unroll_v5(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int W_tail = W % 8; 
    const int W_unrolled = W - W_tail;
    const int W_deg = 8;
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W_unrolled; ox+=W_deg) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const int in_row = (oy + ky) * in_stride;
                const int ker_row = ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const float ker_val = ker[ker_row + kx];
                    acc0 += in[in_row + (ox + kx)] * ker_val;
                    acc1 += in[in_row + (ox + 1 + kx)] * ker_val;
                    acc2 += in[in_row + (ox + 2 + kx)] * ker_val;
                    acc3 += in[in_row + (ox + 3 + kx)] * ker_val;
                    acc4 += in[in_row + (ox + 4 + kx)] * ker_val;
                    acc5 += in[in_row + (ox + 5 + kx)] * ker_val;
                    acc6 += in[in_row + (ox + 6 + kx)] * ker_val;
                    acc7 += in[in_row + (ox + 7 + kx)] * ker_val; 
                }
            }
            out[oy * W + ox] = acc0;
            out[oy * W + ox + 1] = acc1;
            out[oy * W + ox + 2] = acc2;
            out[oy * W + ox + 3] = acc3;
            out[oy * W + ox + 4] = acc4;
            out[oy * W + ox + 5] = acc5;
            out[oy * W + ox + 6] = acc6;
            out[oy * W + ox + 7] = acc7;
        }
    }
}

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    conv_unroll_v5(in, out, ker, H, W, K);
}

// This main function is only used for profiling and is done so with the same 
// compiler optimizations as were there in the makefile. 
// This was necessary as the main.cpp has a lot of overhead such as running naive
// This messes up the readings given by the perf command. 
// The command used to run this 
//g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall -DSTANDALONE_TEST src/conv_unroll.cpp -o bin/conv_prof_unroll
// We will offset the bias introduced by the random generation by having a run in which no conv function is called
// This can be used to offset the random generation misses.
#ifdef STANDALONE_TEST
int main(int argc, char** argv) {
    std::printf("PROFILING TILING.\n");
    int H = 2048, W = 2048, K = 3;
    unsigned seed = 1234;
    if (argc >= 4) {
        H = std::atoi(argv[1]);
        W = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
        std::printf("Setting H=%d, W=%d, K=%d\n",H,W,K);
    }
    if (argc >= 6) seed = static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10));
    float* img = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    float* ker = pa1::alloc_floats(static_cast<std::size_t>(K) * K);
    float* out = pa1::alloc_floats(static_cast<std::size_t>(H) * W);

    pa1::fill_random(img, static_cast<std::size_t>(H) * W, seed);
    pa1::fill_random(ker, static_cast<std::size_t>(K) * K, seed + 1u);
    float* in = pa1::make_padded(img, H, W, K);  // zero-padded halo buffer, stride W+2p
    conv_unroll_v5(in,out,ker,H,W,K);
    return 0;
}
#endif