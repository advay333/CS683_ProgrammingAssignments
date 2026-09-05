#include <cstdio> // For profiling purposes explained below
#include "utils.h" // For profiling purposes explained below

// This main function is only used for profiling and is done so with the same 
// compiler optimizations as were there in the makefile. 
// This was necessary as the main.cpp has a lot of overhead such as running naive
// This messes up the readings given by the perf command. 
// The command used to run this 
//g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall -DSTANDALONE_TEST src/perf_caliberation.cpp -o bin/perf_caliberation
// This can be used to offset the random generation misses from the other runs
int main(int argc, char** argv) {
    std::printf("CALIBERATION RUN FOR RANDOM MATRIX GENERATION.\n");
    int H = 2048, W = 2048, K = 3;
    unsigned seed = 1234;
    if (argc >= 4) {
        H = std::atoi(argv[1]);
        W = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
        std::printf("Setting H=%d, W=%d, K=%d\n",H,W,K);
        seed = static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10));
    }
    float* img = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    float* ker = pa1::alloc_floats(static_cast<std::size_t>(K) * K);
    float* out = pa1::alloc_floats(static_cast<std::size_t>(H) * W);

    pa1::fill_random(img, static_cast<std::size_t>(H) * W, seed);
    pa1::fill_random(ker, static_cast<std::size_t>(K) * K, seed + 1u);
    float* in = pa1::make_padded(img, H, W, K);  // zero-padded halo buffer, stride W+2p
    out[0]=1.0;//To avoid unused optimization
    in[0]=1.0;//To avoid unused optimization
    return 0;
}