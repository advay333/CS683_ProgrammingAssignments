// profile_main.cpp  PARAMETER-SWEEP DRIVER (companion to main.cpp, which is untouched).
//
// Why this exists: main.cpp re-runs the naive reference on every invocation (once for the
// correctness reference, then 1 warmup + 3 timed reps). At M=N=K=2048 that is ~85 s of
// naive work per invocation, and it is identical for every (a_degree, b_degree) point in
// the sweep. This driver takes the naive baseline as a command-line argument so it is
// measured once per (size, seed) for the whole sweep instead of 168 times.
//
// Everything else matches main.cpp: same allocator, same RNG fill, same contiguous strides
// (lda=K, ldb=K, ldc=N), same relative tolerance, same timer, and it is compiled with the
// exact same CXXFLAGS via the `profile` target in the Makefile.
//
// Usage:
//   matmul_profile <variant> <M> <N> <K> <seed> <samples> [baseline_ms] [verify]
//
//   variant     : naive | v1 | tiled | tiled_simd | dispatch
//                 (dispatch = whatever matmul_prefetch() currently forwards to)
//   samples     : number of INDEPENDENT timed runs; each is reported as its own row, no
//                 averaging is done here (average offline from the CSV).
//   baseline_ms : naive time in ms for this workload. <= 0 (or omitted) -> measure it here.
//   verify      : 1 (default) -> check the result against a naive reference; 0 -> skip.
//
// stdout carries a human-readable block plus machine-parsable lines beginning with "CSV,"
// that the sweep script appends straight into results.csv.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "matmul.h"
#include "timer.h"
#include "utils.h"

// The three prefetch variants live in matmul_prefetch.cpp and are not declared in
// matmul.h, so declare them here. Signatures must match MatMulFn exactly.
void matmul_prefetch_v1(const float* A, const float* B, float* C,
                        int M, int N, int K, int lda, int ldb, int ldc);
void matmul_prefetch_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K, int lda, int ldb, int ldc);
void matmul_prefetch_tiled_simd(const float* A, const float* B, float* C,
                                int M, int N, int K, int lda, int ldb, int ldc);

// Same tolerance as the graded harness.
static constexpr float kRelTol = 1e-4f;

struct Variant {
    const char* key;
    MatMulFn fn;
};

static const Variant kVariants[] = {
    {"naive", matmul_naive},
    {"v1", matmul_prefetch_v1},
    {"tiled", matmul_prefetch_tiled},
    {"tiled_simd", matmul_prefetch_tiled_simd},
    {"dispatch", matmul_prefetch},
};
static constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

static void usage(const char* prog) {
    std::printf("Usage: %s <variant> <M> <N> <K> <seed> <samples> [baseline_ms] [verify]\n",
                prog);
    std::printf("variant: ");
    for (int i = 0; i < kNumVariants; ++i)
        std::printf("%s%s", kVariants[i].key, i + 1 < kNumVariants ? " | " : "\n");
}

static MatMulFn lookup(const char* key) {
    for (int i = 0; i < kNumVariants; ++i)
        if (std::strcmp(key, kVariants[i].key) == 0) return kVariants[i].fn;
    return nullptr;
}

static double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        usage(argv[0]);
        return 1;
    }

    const char* variant = argv[1];
    MatMulFn fn = lookup(variant);
    if (!fn) {
        std::printf("error: unknown variant '%s'\n", variant);
        usage(argv[0]);
        return 1;
    }

    const int M = std::atoi(argv[2]);
    const int N = std::atoi(argv[3]);
    const int K = std::atoi(argv[4]);
    const unsigned seed = static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10));
    const int samples = std::atoi(argv[6]);
    const double baseline_arg = (argc >= 8) ? std::atof(argv[7]) : 0.0;
    const bool verify = (argc >= 9) ? (std::atoi(argv[8]) != 0) : true;

    if (M <= 0 || N <= 0 || K <= 0 || samples <= 0) {
        std::printf("error: M, N, K and samples must be positive.\n");
        return 1;
    }

    const bool is_naive = (fn == matmul_naive);

    float* A = pa1::alloc_floats(static_cast<std::size_t>(M) * K);
    float* B = pa1::alloc_floats(static_cast<std::size_t>(N) * K);
    float* C = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
    float* ref = nullptr;

    pa1::fill_random(A, static_cast<std::size_t>(M) * K, seed);
    pa1::fill_random(B, static_cast<std::size_t>(N) * K, seed + 1u);

    // Contiguous microbenchmark strides, identical to main.cpp.
    const int lda = K, ldb = K, ldc = N;
    const double flops = pa1::matmul_flops(M, N, K);

    // ---- correctness reference (skipped for the naive variant, which IS the reference) --
    int correct = -1;  // -1 = not checked
    float tol = 0.0f;
    if (verify && !is_naive) {
        ref = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
        matmul_naive(A, B, ref, M, N, K, lda, ldb, ldc);
        const float ref_mag = pa1::max_abs(ref, static_cast<std::size_t>(M) * N);
        tol = kRelTol * (ref_mag + 1e-30f);
    }

    auto run = [&]() { fn(A, B, C, M, N, K, lda, ldb, ldc); };

    // One untimed run: warms the caches and produces the result we check.
    run();
    if (verify && !is_naive) {
        correct = (pa1::max_abs_diff(C, ref, static_cast<std::size_t>(M) * N) <= tol) ? 1 : 0;
    }

    // ---- timed samples: each is an independent measurement, reported separately --------
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(samples));
    for (int s = 0; s < samples; ++s) {
        times.push_back(pa1::time_median_ms(run, /*warmup=*/0, /*reps=*/1));
    }

    const double baseline_ms = (baseline_arg > 0.0) ? baseline_arg
                             : (is_naive ? median_of(times) : 0.0);

    std::printf("variant=%s M=%d N=%d K=%d seed=%u samples=%d verify=%s correct=%s\n",
                variant, M, N, K, seed, samples, verify ? "yes" : "no",
                (correct < 0) ? "n/a" : (correct ? "yes" : "NO"));
    if (baseline_ms > 0.0) std::printf("baseline_ms=%.6f\n", baseline_ms);
    std::printf("%-8s  %12s  %10s  %9s\n", "sample", "time(ms)", "GFLOP/s", "speedup");
    std::printf("----------------------------------------------\n");

    for (int s = 0; s < samples; ++s) {
        const double ms = times[static_cast<std::size_t>(s)];
        const double gflops = (ms > 0.0) ? flops / (ms * 1e6) : 0.0;
        const double speedup = (baseline_ms > 0.0 && ms > 0.0) ? baseline_ms / ms : 0.0;
        std::printf("%-8d  %12.4f  %10.2f  %8.2fx\n", s + 1, ms, gflops, speedup);
        // Machine-parsable row. Column order is fixed; the sweep script writes the header.
        std::printf("CSV,%s,%d,%d,%d,%u,%d,%.6f,%.4f,%.6f,%.6f,%s\n", variant, M, N, K, seed,
                    s + 1, ms, gflops, baseline_ms, speedup,
                    (correct < 0) ? "na" : (correct ? "yes" : "no"));
    }

    // Median line for the naive variant so the sweep script can harvest the baseline.
    if (is_naive) std::printf("BASELINE_MS %.6f\n", median_of(times));

    pa1::free_floats(A);
    pa1::free_floats(B);
    pa1::free_floats(C);
    if (ref) pa1::free_floats(ref);
    return 0;
}