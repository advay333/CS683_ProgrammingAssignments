// // conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// // Hint: measure after every change. Not every "optimization" helps  let the numbers,
// // not intuition, decide.

// #include <immintrin.h>

// #include "convolution.h"

// void conv_optimized(const float* in, float* out, const float* ker,
//                     int H, int W, int K) {
//     // TODO(student): replace this placeholder with your best combined implementation.
//     conv_naive(in, out, ker, H, W, K);
// }

// conv_optimized.cpp  STAGE 5: ALL TECHNIQUES COMBINED
//
// The four earlier stages are not four independent multipliers. They are four levels
// of ONE blocking hierarchy, each matched to a level of storage:
//
//   YMM lanes      <- SIMD over ox                (8 pixels per instruction)
//   register file  <- register blocking U x V     (U=4 vectors wide, V=2 rows tall)
//   L1 / L2        <- column panels of width PW   (tiling: shortens the reuse distance)
//   DRAM           <- row-major order + NT stores (removes the read-for-ownership)
//
// Why the pieces compose the way they do:
//
//  * REORDER: its value is that ker[ky*K+kx] becomes a loop-invariant scalar and the
//    inner sweep is unit stride -- exactly the shape a broadcast-FMA wants.  Its cost,
//    in conv_reorder.cpp, is that out[] is the accumulator, so the kernel makes K^2
//    full passes over memory (~440 MB instead of ~34 MB).  Here we keep the loop order
//    and hold the accumulators in REGISTERS, so we get the benefit and none of the cost.
//
//  * UNROLL: naive chains all K^2 FMAs through one accumulator.  FMA latency is ~4
//    cycles and throughput is 2/cycle, so a single chain runs at 1/8 of peak.  U*V
//    independent accumulators break the chain and, at the same time, let one loaded
//    vector feed several outputs.
//
//  * SIMD: 9 vector FMAs produce 8 pixels, at 2 FMA/cycle -> 0.5625 cycles/pixel,
//    i.e. 32 FLOP/cycle, which is the hardware peak.  Nothing that computes all K^2
//    taps can beat that, so this is the compute floor.
//
//  * TILING: the reuse distance for the halo rows is (V+K-1)*W*4 bytes ~= 33 KB at
//    K=3 and ~49 KB at K=5, both far inside a 1.25 MiB per-core L2.  Panelling is
//    therefore implemented and tunable but deliberately wide -- narrow tiles (e.g.
//    32x32) are actively harmful: 32 floats is a 128-byte run, too short for the L2
//    streaming prefetcher, and they add (K-1)/PW redundant traffic.
//
//  * NT STORES: at 2048x2048 the working set is 33.6 MB, past L3, so the kernel is
//    DRAM-bound once the above are in place.  An ordinary store first READS the
//    destination cache line (read-for-ownership) even though we overwrite all 64
//    bytes; _mm256_stream_ps skips that, cutting traffic from ~50 MB to ~34 MB.
//    This only pays off AFTER SIMD + blocking have moved the bottleneck to memory --
//    the clearest superadditive effect in the whole exercise.
//
// IMPLEMENTATION NOTE (measured, not assumed): the accumulators are written as named
// variables rather than as __m256 arrays.  At -O2 (no -O3) GCC does not reliably
// scalarize an array of vectors, and the spilled version measured ~2x slower.
//
// Builds under the pinned flags: -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma

#include "convolution.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace {

// ------------------------------------------------------------------ tuning knobs
// Column-panel width (the tiling level), rounded down to a multiple of 32.
// Set >= any W to disable panelling entirely.  Measured: wide is better.
constexpr int kPanelW = 1 << 20;

// Register block: 4 vectors (32 pixels) wide x 2 output rows tall = 8 accumulators.
constexpr int kBlockW = 32;  // 4 * 8 pixels
constexpr int kBlockH = 2;   // V

// --------------------------------------------------------------------- utilities

inline bool aligned32(const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) & std::uintptr_t{31}) == 0;
}

// Streaming store when it is legal, ordinary store otherwise.  NT is a template
// parameter, so the branch is resolved at compile time and costs nothing.
template <bool NT>
inline void st8(float* p, __m256 v) {
    if (NT) {
        _mm256_stream_ps(p, v);
    } else {
        _mm256_storeu_ps(p, v);
    }
}

// Scalar fallback: leftover columns, and the last output row when H is odd.
inline void scalar_run(const float* in, float* out, const float* ker, int oy, int xa,
                       int xb, int W, int in_stride, int K) {
    for (int x = xa; x < xb; ++x) {
        float acc = 0.0f;
        for (int ky = 0; ky < K; ++ky) {
            const float* r = in + static_cast<std::size_t>(oy + ky) * in_stride + x;
            const float* k = ker + ky * K;
            for (int kx = 0; kx < K; ++kx) acc += r[kx] * k[kx];
        }
        out[static_cast<std::size_t>(oy) * W + x] = acc;
    }
}

// ---------------------------------------------------- K == 3 hot block (32 x 2)
//
// Input row -> output row map for V = 2 (r = row index within the block):
//   r0 -> top row,    kernel row 0
//   r1 -> top row,    kernel row 1   AND   bottom row, kernel row 0
//   r2 -> top row,    kernel row 2   AND   bottom row, kernel row 1
//   r3 ->                                  bottom row, kernel row 2
// The two middle rows are shared, so each of their vectors is loaded ONCE and used
// TWICE.  That is what keeps the block FMA-bound (36 cycles of FMA against ~22
// cycles of load-port work) instead of load-bound.
//
// Per block: 48 loads, 72 FMAs, 8 stores, 64 pixels -> 0.5625 cycles/pixel.
template <bool NT>
inline void k3_block(const float* r0, const float* r1, const float* r2, const float* r3,
                     float* o0, float* o1, int x, const __m256 (&k)[3][3]) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    __m256 b0 = a0, b1 = a0, b2 = a0, b3 = a0;

#define LD(P, O) _mm256_loadu_ps((P) + x + (O))

    // r0: top output row only.
    for (int j = 0; j < 3; ++j) {
        a0 = _mm256_fmadd_ps(k[0][j], LD(r0, j + 0), a0);
        a1 = _mm256_fmadd_ps(k[0][j], LD(r0, j + 8), a1);
        a2 = _mm256_fmadd_ps(k[0][j], LD(r0, j + 16), a2);
        a3 = _mm256_fmadd_ps(k[0][j], LD(r0, j + 24), a3);
    }
    // r1: shared -- one load, two FMAs.
    for (int j = 0; j < 3; ++j) {
        __m256 v;
        v = LD(r1, j + 0);  a0 = _mm256_fmadd_ps(k[1][j], v, a0); b0 = _mm256_fmadd_ps(k[0][j], v, b0);
        v = LD(r1, j + 8);  a1 = _mm256_fmadd_ps(k[1][j], v, a1); b1 = _mm256_fmadd_ps(k[0][j], v, b1);
        v = LD(r1, j + 16); a2 = _mm256_fmadd_ps(k[1][j], v, a2); b2 = _mm256_fmadd_ps(k[0][j], v, b2);
        v = LD(r1, j + 24); a3 = _mm256_fmadd_ps(k[1][j], v, a3); b3 = _mm256_fmadd_ps(k[0][j], v, b3);
    }
    // r2: shared.
    for (int j = 0; j < 3; ++j) {
        __m256 v;
        v = LD(r2, j + 0);  a0 = _mm256_fmadd_ps(k[2][j], v, a0); b0 = _mm256_fmadd_ps(k[1][j], v, b0);
        v = LD(r2, j + 8);  a1 = _mm256_fmadd_ps(k[2][j], v, a1); b1 = _mm256_fmadd_ps(k[1][j], v, b1);
        v = LD(r2, j + 16); a2 = _mm256_fmadd_ps(k[2][j], v, a2); b2 = _mm256_fmadd_ps(k[1][j], v, b2);
        v = LD(r2, j + 24); a3 = _mm256_fmadd_ps(k[2][j], v, a3); b3 = _mm256_fmadd_ps(k[1][j], v, b3);
    }
    // r3: bottom output row only.
    for (int j = 0; j < 3; ++j) {
        b0 = _mm256_fmadd_ps(k[2][j], LD(r3, j + 0), b0);
        b1 = _mm256_fmadd_ps(k[2][j], LD(r3, j + 8), b1);
        b2 = _mm256_fmadd_ps(k[2][j], LD(r3, j + 16), b2);
        b3 = _mm256_fmadd_ps(k[2][j], LD(r3, j + 24), b3);
    }
#undef LD

    st8<NT>(o0 + x + 0, a0);  st8<NT>(o0 + x + 8, a1);
    st8<NT>(o0 + x + 16, a2); st8<NT>(o0 + x + 24, a3);
    st8<NT>(o1 + x + 0, b0);  st8<NT>(o1 + x + 8, b1);
    st8<NT>(o1 + x + 16, b2); st8<NT>(o1 + x + 24, b3);
}

// ------------------------------------------ generic odd-K hot block (32 x 2)
// Same structure with runtime K: V+K-1 distinct input rows feed V*K (row, kernel-row)
// pairs, so each loaded vector is reused up to min(V, K) times.
template <bool NT>
inline void kg_block(const float* rbase, int in_stride, float* o0, float* o1,
                     const float* ker, int K, int x) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    __m256 b0 = a0, b1 = a0, b2 = a0, b3 = a0;

    const int nrows = K + 1;  // V + K - 1 with V = 2
    for (int r = 0; r < nrows; ++r) {
        const float* p = rbase + static_cast<std::size_t>(r) * in_stride + x;
        const bool useA = (r < K);       // kernel row r feeds the top output row
        const bool useB = (r >= 1);      // kernel row r-1 feeds the bottom output row
        const float* ka = useA ? ker + r * K : nullptr;
        const float* kb = useB ? ker + (r - 1) * K : nullptr;

        for (int j = 0; j < K; ++j) {
            const __m256 va = useA ? _mm256_set1_ps(ka[j]) : _mm256_setzero_ps();
            const __m256 vb = useB ? _mm256_set1_ps(kb[j]) : _mm256_setzero_ps();
            __m256 v;
            v = _mm256_loadu_ps(p + j + 0);
            if (useA) a0 = _mm256_fmadd_ps(va, v, a0);
            if (useB) b0 = _mm256_fmadd_ps(vb, v, b0);
            v = _mm256_loadu_ps(p + j + 8);
            if (useA) a1 = _mm256_fmadd_ps(va, v, a1);
            if (useB) b1 = _mm256_fmadd_ps(vb, v, b1);
            v = _mm256_loadu_ps(p + j + 16);
            if (useA) a2 = _mm256_fmadd_ps(va, v, a2);
            if (useB) b2 = _mm256_fmadd_ps(vb, v, b2);
            v = _mm256_loadu_ps(p + j + 24);
            if (useA) a3 = _mm256_fmadd_ps(va, v, a3);
            if (useB) b3 = _mm256_fmadd_ps(vb, v, b3);
        }
    }

    st8<NT>(o0 + x + 0, a0);  st8<NT>(o0 + x + 8, a1);
    st8<NT>(o0 + x + 16, a2); st8<NT>(o0 + x + 24, a3);
    st8<NT>(o1 + x + 0, b0);  st8<NT>(o1 + x + 8, b1);
    st8<NT>(o1 + x + 16, b2); st8<NT>(o1 + x + 24, b3);
}

// ------------------------- narrow tail block: 8 pixels x 2 rows, any odd K --------
template <bool NT>
inline void kg_block8(const float* rbase, int in_stride, float* o0, float* o1,
                      const float* ker, int K, int x) {
    __m256 a0 = _mm256_setzero_ps(), b0 = a0;
    for (int r = 0; r < K + 1; ++r) {
        const float* p = rbase + static_cast<std::size_t>(r) * in_stride + x;
        const bool useA = (r < K), useB = (r >= 1);
        for (int j = 0; j < K; ++j) {
            const __m256 v = _mm256_loadu_ps(p + j);
            if (useA) a0 = _mm256_fmadd_ps(_mm256_set1_ps(ker[r * K + j]), v, a0);
            if (useB) b0 = _mm256_fmadd_ps(_mm256_set1_ps(ker[(r - 1) * K + j]), v, b0);
        }
    }
    st8<NT>(o0 + x, a0);
    st8<NT>(o1 + x, b0);
}

inline int panel_width(int W) {
    long pw = (static_cast<long>(kPanelW) / kBlockW) * kBlockW;
    if (pw < kBlockW) pw = kBlockW;
    if (pw > W) pw = W;
    return static_cast<int>(pw);
}

// -------------------------------------------------------------- driver, K == 3
template <bool NT>
void run_k3(const float* __restrict in, float* __restrict out, const float* __restrict ker,
            int H, int W, int in_stride) {
    __m256 k[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) k[i][j] = _mm256_set1_ps(ker[i * 3 + j]);

    const int PW = panel_width(W);

    for (int x0 = 0; x0 < W; x0 += PW) {                // TILING: column panel, outer
        const int x1 = (x0 + PW < W) ? x0 + PW : W;

        int oy = 0;
        for (; oy + kBlockH <= H; oy += kBlockH) {      // V-row register block
            const float* r0 = in + static_cast<std::size_t>(oy) * in_stride;
            const float* r1 = r0 + in_stride;
            const float* r2 = r1 + in_stride;
            const float* r3 = r2 + in_stride;
            float* o0 = out + static_cast<std::size_t>(oy) * W;
            float* o1 = o0 + W;

            int x = x0;
            for (; x + kBlockW <= x1; x += kBlockW)
                k3_block<NT>(r0, r1, r2, r3, o0, o1, x, k);
            for (; x + 8 <= x1; x += 8)
                kg_block8<NT>(r0, in_stride, o0, o1, ker, 3, x);
            if (x < x1) {
                scalar_run(in, out, ker, oy, x, x1, W, in_stride, 3);
                scalar_run(in, out, ker, oy + 1, x, x1, W, in_stride, 3);
            }
        }
        for (; oy < H; ++oy)                            // odd trailing output row
            scalar_run(in, out, ker, oy, x0, x1, W, in_stride, 3);
    }
}

// -------------------------------------------------------- driver, generic odd K
template <bool NT>
void run_gen(const float* __restrict in, float* __restrict out, const float* __restrict ker,
             int H, int W, int in_stride, int K) {
    const int PW = panel_width(W);

    for (int x0 = 0; x0 < W; x0 += PW) {
        const int x1 = (x0 + PW < W) ? x0 + PW : W;

        int oy = 0;
        for (; oy + kBlockH <= H; oy += kBlockH) {
            const float* rbase = in + static_cast<std::size_t>(oy) * in_stride;
            float* o0 = out + static_cast<std::size_t>(oy) * W;
            float* o1 = o0 + W;

            int x = x0;
            for (; x + kBlockW <= x1; x += kBlockW)
                kg_block<NT>(rbase, in_stride, o0, o1, ker, K, x);
            for (; x + 8 <= x1; x += 8)
                kg_block8<NT>(rbase, in_stride, o0, o1, ker, K, x);
            if (x < x1) {
                scalar_run(in, out, ker, oy, x, x1, W, in_stride, K);
                scalar_run(in, out, ker, oy + 1, x, x1, W, in_stride, K);
            }
        }
        for (; oy < H; ++oy) scalar_run(in, out, ker, oy, x0, x1, W, in_stride, K);
    }
}

}  // namespace

// ------------------------------------------------------------------- entry point
void conv_optimized(const float* in, float* out, const float* ker, int H, int W, int K) {
    if (H <= 0 || W <= 0 || K <= 0) return;

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // Streaming stores require every destination address to be 32B aligned.
    // W % 8 == 0 implies W*4 % 32 == 0, so every row start shares the base
    // alignment and every block offset is a multiple of 8 floats.  Verify, do not
    // assume -- a misaligned _mm256_stream_ps faults.
    const bool nt_ok = aligned32(out) && (W % 8 == 0);

    if (K == 3) {
        if (nt_ok) run_k3<true>(in, out, ker, H, W, in_stride);
        else       run_k3<false>(in, out, ker, H, W, in_stride);
    } else {
        if (nt_ok) run_gen<true>(in, out, ker, H, W, in_stride, K);
        else       run_gen<false>(in, out, ker, H, W, in_stride, K);
    }

    // NT stores are weakly ordered: fence before the caller reads out[].
    if (nt_ok) _mm_sfence();
}