// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
#include <algorithm> // for std::min
#include <cstdio> // For profiling purposes explained below
#include "utils.h" // For profiling purposes explained below

int tile_size=1024;

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    // conv_naive(in, out, ker, H, W, K);
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    
    const int tile_x=tile_size;
    const int tile_y=tile_x;
    //std::printf("Using tile size %d", tile_x);
    for (int oy = 0; oy < H; oy+=tile_y) {
        for (int ox = 0; ox < W; ox+=tile_x) {
            const int bound_y = std::min(tile_y+oy,H);
            const int bound_x = std::min(tile_x+ox,W);
            for (int ty = oy; ty < bound_y; ty++) {
                for (int tx = ox; tx < bound_x; tx++) {         
                    float acc = 0.0f;
                    const float* in_point= &in[ty*in_stride + tx];
                    float* out_point=&out[ty*W+tx];
                    for (int ky = 0; ky < K; ++ky) {
                        const int in_row=ky*in_stride;
                        const int k_row=ky*K;
                        for (int kx = 0; kx < K; ++kx) {
                            acc += in_point[in_row + kx] * ker[k_row + kx];
                        }
                    }
                    *out_point = acc;
                }
            }
        }
    } 
}

// This main function is only used for profiling and is done so with the same 
// compiler optimizations as were there in the makefile. 
// This was necessary as the main.cpp has a lot of overhead such as running naive
// This messes up the readings given by the perf command. 
// The command used to run this 
//g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall -DSTANDALONE_TEST src/conv_tile.cpp -o bin/conv_prof_tile
// We will offset the bias introduced by the random generation by having a run in which no conv function is called
// This can be used to offset the random generation misses.
#ifdef STANDALONE_TEST
#include <cstdlib>
#include <cstring>

// Standalone profiling driver.
//
//   usage: <prog> [H [W [K [seed [variant]]]]]
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
                "tile", variant, H, W, K, seed);

    // For this stage the variant string is the tile size in elements;
    // "default" leaves the value compiled into the file untouched.
    if (std::strcmp(variant, "default") != 0) {
        int t = std::atoi(variant);
        if (t <= 0) {
            std::fprintf(stderr, "bad tile size '%s'\n", variant);
            return 2;
        }
        tile_size = t;
    }
    std::printf("tile_size=%d\n", tile_size);

    float* img = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    float* ker = pa1::alloc_floats(static_cast<std::size_t>(K) * K);
    float* out = pa1::alloc_floats(static_cast<std::size_t>(H) * W);

    pa1::fill_random(img, static_cast<std::size_t>(H) * W, seed);
    pa1::fill_random(ker, static_cast<std::size_t>(K) * K, seed + 1u);
    float* in = pa1::make_padded(img, H, W, K);  // zero-padded halo buffer, stride W+2p

    conv_tile(in, out, ker, H, W, K);

    std::printf("checksum %.6f\n",
                static_cast<double>(out[0])
                    + static_cast<double>(out[static_cast<std::size_t>(H) * W - 1])
                    + static_cast<double>(in[0]));
    return 0;
}
#endif