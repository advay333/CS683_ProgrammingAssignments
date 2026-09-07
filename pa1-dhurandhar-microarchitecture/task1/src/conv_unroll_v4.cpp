// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"
#include <cstdio> // For profiling purposes explained below
#include "utils.h" // For profiling purposes explained below


void conv_unroll_v4(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int W_tail = W % 2; 
    const int W_unrolled = W - W_tail;
    const int H_tail = H % 2;
    const int H_unrolled = H - H_tail;

    for (int oy = 0; oy < H_unrolled; oy+=2) {
        
        for (int ox = 0; ox < W_unrolled; ox+=2) {
            float acc00 = 0.0f;
            float acc01 = 0.0f;
            float acc10 = 0.0f;
            float acc11 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const int in_row = (oy + ky) * in_stride;
                const int in_row_next = (oy + 1 + ky) * in_stride;
                const int ker_row = ky * K;
                
                for (int kx = 0; kx < K-1; kx += 2) {
                    const float ker_val = ker[ker_row + kx];
                    const float ker_val_next = ker[ker_row + kx + 1];
                    acc00 += in[in_row + (ox + kx)] * ker_val;
                    acc00 += in[in_row + (ox + kx + 1)] * ker_val_next;

                    acc01 += in[in_row + (ox + 1 + kx)] * ker_val;
                    acc01 += in[in_row + (ox + 1 + kx + 1)] * ker_val_next;

                    acc10 += in[in_row_next + (ox + kx)] * ker_val;
                    acc10 += in[in_row_next + (ox + kx + 1)] * ker_val_next;
                    acc11 += in[in_row_next + (ox + 1 + kx)] * ker_val;
                    acc11 += in[in_row_next + (ox + 1 + kx + 1)] * ker_val_next;

                }
                const float ker_val_last = ker[ker_row + (K - 1)];
                acc00 += in[in_row + (ox + K - 1)] * ker_val_last;
                acc01 += in[in_row + (ox + 1 + K - 1)] * ker_val_last;
                acc10 += in[in_row_next + (ox + K - 1)] * ker_val_last;
                acc11 += in[in_row_next + (ox + 1 + K - 1)] * ker_val_last;
            }
            out[oy * W + ox] = acc00;
            out[oy * W + ox + 1] = acc01;
            out[(oy + 1) * W + ox] = acc10;
            out[(oy + 1) * W + ox + 1] = acc11;
        }
        // Handle the tail elements if W is odd
        if (W_tail == 1) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const int in_row = (oy + ky) * in_stride;
                const int in_row_next = (oy + 1 + ky) * in_stride;
                const int ker_row = ky * K;

                for(int kx = 0; kx < K-1; kx += 2) {
                    const float ker_val = ker[ker_row + kx];
                    const float ker_val_next = ker[ker_row + kx + 1];
                    acc0 += in[in_row + (W - 1 + kx)] * ker_val;
                    acc0 += in[in_row + (W - 1 + kx + 1)] * ker_val_next;
                    acc1 += in[in_row_next + (W - 1 + kx)] * ker_val;
                    acc1 += in[in_row_next + (W - 1 + kx + 1)] * ker_val_next;
                }
                const float ker_val_last = ker[ker_row + (K - 1)];
                acc0 += in[in_row + (W - 1 + K - 1)] * ker_val_last;
                acc1 += in[in_row_next + (W - 1 + K - 1)] * ker_val_last;
            }
            out[oy * W + (W - 1)] = acc0;
            out[(oy + 1) * W + (W - 1)] = acc1;
        }
    }
        // Handle the tail elements if H is odd
    if (H_tail == 1) {
        for(int ox = 0; ox < W_unrolled; ox+=2) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const int in_row = (H - 1 + ky) * in_stride;
                const int ker_row = ky * K;

                for(int kx = 0; kx < K-1; kx += 2) {
                    const float ker_val = ker[ker_row + kx];
                    const float ker_val_next = ker[ker_row + kx + 1];
                    acc0 += in[in_row + (ox + kx)] * ker_val;
                    acc0 += in[in_row + (ox + kx + 1)] * ker_val_next;
                    acc1 += in[in_row + (ox + 1 + kx)] * ker_val;
                    acc1 += in[in_row + (ox + 1 + kx + 1)] * ker_val_next;
                }
                const float ker_val_last = ker[ker_row + (K - 1)];
                acc0 += in[in_row + (ox + K - 1)] * ker_val_last;
                acc1 += in[in_row + (ox + 1 + K - 1)] * ker_val_last;
            }
            out[(H - 1) * W + ox] = acc0;
            out[(H - 1) * W + ox + 1] = acc1;
        }
        if(W_tail == 1) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const int in_row = (H - 1 + ky) * in_stride;
                const int ker_row = ky * K;
                for(int kx = 0; kx < K-1; kx += 2) {
                    const float ker_val = ker[ker_row + kx];
                    const float ker_val_next = ker[ker_row + kx + 1];
                    acc += in[in_row + (W - 1 + kx)] * ker_val;
                    acc += in[in_row + (W - 1 + kx + 1)] * ker_val_next;
                }
                const float ker_val_last = ker[ker_row + (K - 1)];
                acc += in[in_row + (W - 1 + K - 1)] * ker_val_last;
            }
            out[(H - 1) * W + (W - 1)] = acc;
        }
    }
}

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
    conv_unroll_v4(in,out,ker,H,W,K);
    return 0;
}
#endif