// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    //conv_naive(in, out, ker, H, W, K);
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            out[oy * W + ox] = 0.0f;// Initialisation run 
        }
    } 

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            float ker_val=ker[ky*K+kx];
            for (int oy = 0; oy < H; ++oy) {
                const float* in_index= in + (oy + ky) * in_stride + kx;
                float* out_index= out + oy * W;
                int ox = 0;
                for (; ox < W-8; ox+=8) {
                    out_index[ox] += in_index[ox] * ker_val;
                    out_index[ox+1] += in_index[ox+1] * ker_val;
                    out_index[ox+2] += in_index[ox+2] * ker_val;
                    out_index[ox+3] += in_index[ox+3] * ker_val;                    
                    out_index[ox+4] += in_index[ox+4] * ker_val;
                    out_index[ox+5] += in_index[ox+5] * ker_val;
                    out_index[ox+6] += in_index[ox+6] * ker_val;
                    out_index[ox+7] += in_index[ox+7] * ker_val; 
                }
                for(; ox < W ; ox++){
                    out_index[ox] += in_index[ox] * ker_val;
                }
            }
        }
    }
}
