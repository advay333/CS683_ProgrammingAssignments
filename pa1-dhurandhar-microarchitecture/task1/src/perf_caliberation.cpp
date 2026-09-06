#include <cstdio> // For profiling purposes explained below
#include "utils.h" // For profiling purposes explained below

// This main function is only used for profiling and is done so with the same 
// compiler optimizations as were there in the makefile. 
// This was necessary as the main.cpp has a lot of overhead such as running naive
// This messes up the readings given by the perf command. 
// The command used to run this 
//g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall -DSTANDALONE_TEST src/perf_caliberation.cpp -o bin/perf_caliberation
// This can be used to offset the random generation misses from the other runs

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
                "calibration", variant, H, W, K, seed);

    float* img = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    float* ker = pa1::alloc_floats(static_cast<std::size_t>(K) * K);
    float* out = pa1::alloc_floats(static_cast<std::size_t>(H) * W);

    pa1::fill_random(img, static_cast<std::size_t>(H) * W, seed);
    pa1::fill_random(ker, static_cast<std::size_t>(K) * K, seed + 1u);
    float* in = pa1::make_padded(img, H, W, K);  // zero-padded halo buffer, stride W+2p

    // No convolution here: this run reproduces exactly the allocation,
    // RNG fill and halo-padding work that every other binary performs
    // before its kernel starts, so it can be used as an offset later.
    (void)variant;

    std::printf("checksum %.6f\n",
                static_cast<double>(out[0])
                    + static_cast<double>(out[static_cast<std::size_t>(H) * W - 1])
                    + static_cast<double>(in[0]));
    return 0;
}
#endif