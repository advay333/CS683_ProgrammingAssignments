// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
//
// This file is built up the same way the earlier stages were: one idea at a time,
// each version starting from the previous one, so that the effect of every single
// technique can be measured on its own. Nothing here is new machinery -- every
// version below is made only out of the four things we already did:
//
//   reorder : make the kernel value a loop-invariant scalar and sweep ox innermost
//   unroll  : several independent accumulators to break the FMA dependency chain
//   simd    : 8 outputs per instruction with a broadcast kernel value + FMA
//   tile    : work on a block of the image at a time so reuse happens in cache
//
// The point of the stage is that these are NOT four independent multipliers. They
// are four levels of the SAME blocking idea, one per level of storage:
//
//   YMM register lane  <- simd    (8 pixels)
//   register file      <- unroll  (4 vectors wide x 2 rows tall = 8 accumulators)
//   L1 / L2            <- tile    (column panel: the halo rows stay resident)
//   memory order       <- reorder (unit stride sweeps, prefetcher friendly)
//
// Read the versions in order; each comment says what was added and why.

#include <immintrin.h>

#include "convolution.h"
#include <cstdio>  // For profiling purposes explained below
#include "utils.h" // For profiling purposes explained below

// ---------------------------------------------------------------------------
// Shared scalar helper: computes out[oy][xa..xb) the naive way.
// Used for leftover columns and for the last output row when H is odd, so that
// the vector versions below never have to worry about the edges.
// ---------------------------------------------------------------------------
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

// ===========================================================================
// v1 : SIMD + UNROLL along ox   (this is really conv_simd256_v3 again)
// ---------------------------------------------------------------------------
// Starting point. Worth restating explicitly, because it already contains two
// of the four techniques and it is the honest baseline that the later versions
// have to beat:
//   - simd   : one __m256 holds 8 output pixels
//   - unroll : 4 independent accumulators, so 4 FMAs are in flight at once and
//              the ~4 cycle FMA latency is hidden (throughput is 2 FMA/cycle)
//   - reorder: ker_val is broadcast once and reused by all 4 accumulators
// One thing it does NOT do: every input row is loaded once per output row, even
// though neighbouring output rows need almost the same rows. That is what v2 fixes.
// ===========================================================================
void conv_optimized_v1(const float* in, float* out, const float* ker,
                       int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int BW = 32;              // 4 vectors x 8 pixels
    const int W_blocked = (W / BW) * BW;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W_blocked; ox += BW) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const int in_offset = (oy + ky) * in_stride + ox;
                const int ker_offset = ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 ker_val = _mm256_set1_ps(ker[ker_offset + kx]);
                    acc0 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx]), acc0);
                    acc1 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 8]), acc1);
                    acc2 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 16]), acc2);
                    acc3 = _mm256_fmadd_ps(ker_val, _mm256_loadu_ps(&in[in_offset + kx + 24]), acc3);
                }
            }
            _mm256_storeu_ps(&out[oy * W + ox], acc0);
            _mm256_storeu_ps(&out[oy * W + ox + 8], acc1);
            _mm256_storeu_ps(&out[oy * W + ox + 16], acc2);
            _mm256_storeu_ps(&out[oy * W + ox + 24], acc3);
        }
        for (int ox = W_blocked; ox < W; ++ox) {
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

// ===========================================================================
// v2 : v1 + UNROLL along oy as well  (2D register block, 4 x 2)
// ---------------------------------------------------------------------------
// This is the step conv_unroll_v3 took in scalar form (2x2 block), applied on
// top of the vector version. We now produce TWO output rows at a time.
//
// Why it is not just "more of the same": output rows oy and oy+1 read overlapping
// input rows. For V = 2 output rows and kernel height K, the block touches K+1
// distinct input rows, and K-1 of them are needed by BOTH output rows. So we load
// each of those vectors ONCE and issue TWO FMAs with it.
//
//   K = 3, V = 2:  4 input rows feed 6 (row, kernel-row) pairs
//     r0 -> top only        (kernel row 0)
//     r1 -> top (row 1) AND bottom (row 0)
//     r2 -> top (row 2) AND bottom (row 1)
//     r3 -> bottom only     (kernel row 2)
//
// Load traffic per output pixel drops by ~1/3 while the FMA count is unchanged,
// which moves the block from load-port-bound towards FMA-bound. 8 accumulators
// (4 wide x 2 tall) still fit comfortably in the 16 YMM registers.
//
// The useA/useB flags below are just the row map above written for general odd K.
// ===========================================================================
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

// ===========================================================================
// v3 : v2 + hoist the kernel broadcasts out of the ox loop
// ---------------------------------------------------------------------------
// Pure REORDER thinking, one level further than stage 1. In v2 the broadcast
// _mm256_set1_ps(ker[...]) sits inside the innermost block and is therefore
// re-executed for every 32-pixel block, i.e. W/32 * H/2 * (K+1) * K times, even
// though the kernel never changes. The kernel is at most K*K = 9 or 25 values;
// broadcasting it ONCE per call into an array of __m256 costs nothing and removes
// a load + broadcast uop from the hot loop.
//
// This also frees the FMA to take its multiplier straight from a register, which
// matters because at K=3 the block is only 72 FMAs long -- any extra uop shows up.
// ===========================================================================
static const int kMaxK = 15;   // kernel sizes we pre-broadcast for (K*K <= 225)

void conv_optimized_v3(const float* in, float* out, const float* ker,
                       int H, int W, int K) {
    if (K > kMaxK) {              // absurdly large kernel: v2 is still correct
        conv_optimized_v2(in, out, ker, H, W, K);
        return;
    }
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int BW = 32;
    const int W_blocked = (W / BW) * BW;
    const int H_blocked = (H / 2) * 2;

    // Broadcast the whole kernel once, up front.
    __m256 kv[kMaxK * kMaxK];
    for (int i = 0; i < K * K; ++i) kv[i] = _mm256_set1_ps(ker[i]);

    for (int oy = 0; oy < H_blocked; oy += 2) {
        for (int ox = 0; ox < W_blocked; ox += BW) {
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();
            __m256 b0 = _mm256_setzero_ps();
            __m256 b1 = _mm256_setzero_ps();
            __m256 b2 = _mm256_setzero_ps();
            __m256 b3 = _mm256_setzero_ps();

            for (int r = 0; r < K + 1; ++r) {
                const int in_offset = (oy + r) * in_stride + ox;
                const bool useA = (r < K);
                const bool useB = (r >= 1);

                for (int kx = 0; kx < K; ++kx) {
                    const __m256 v0 = _mm256_loadu_ps(&in[in_offset + kx]);
                    const __m256 v1 = _mm256_loadu_ps(&in[in_offset + kx + 8]);
                    const __m256 v2 = _mm256_loadu_ps(&in[in_offset + kx + 16]);
                    const __m256 v3 = _mm256_loadu_ps(&in[in_offset + kx + 24]);

                    if (useA) {
                        const __m256 ka = kv[r * K + kx];
                        a0 = _mm256_fmadd_ps(ka, v0, a0);
                        a1 = _mm256_fmadd_ps(ka, v1, a1);
                        a2 = _mm256_fmadd_ps(ka, v2, a2);
                        a3 = _mm256_fmadd_ps(ka, v3, a3);
                    }
                    if (useB) {
                        const __m256 kb = kv[(r - 1) * K + kx];
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
        conv_opt_scalar_range(in, out, ker, oy, W_blocked, W, W, in_stride, K);
        conv_opt_scalar_range(in, out, ker, oy + 1, W_blocked, W, W, in_stride, K);
    }
    for (int oy = H_blocked; oy < H; ++oy) {
        conv_opt_scalar_range(in, out, ker, oy, 0, W, W, in_stride, K);
    }
}

// ===========================================================================
// v4 : v3 + TILING, as a column panel
// ---------------------------------------------------------------------------
// The last of the four techniques. In v3 the loop order is (oy, ox), so between
// the moment input row y is read for output row y-1 and the moment it is read
// again for output row y, we have swept a whole image row: the reuse distance is
// about (V + K - 1) * W * 4 bytes. At W = 2048, K = 3 that is ~33 KB -- bigger
// than the 32 KB L1, so those rows come back from L2 every time.
//
// Tiling fixes exactly that: split the image into column panels of width PW and
// walk all the rows of one panel before moving to the next panel. The reuse
// distance becomes (V + K - 1) * PW * 4, which we can make fit in L1.
//
// IMPORTANT (and this is the honest, measured part): the panel must not be made
// narrow just because "tiling is good". Two costs push back:
//   * each panel re-reads a (K-1)-column halo, so traffic grows by (K-1)/PW;
//   * short runs defeat the hardware streaming prefetcher, which wants long
//     sequential runs -- a 32-float panel is only a 128 byte run.
// So PW is a knob to sweep, not a constant to guess. On our machine the K = 3
// working set is already small enough that wide panels (or PW >= W, i.e. tiling
// switched off) measured best; keep the knob and report the sweep.
// ===========================================================================
#ifndef PANEL_W
#define PANEL_W 1048576   // >= any W  ->  tiling effectively off (measured best here)
#endif
static const int kPanelW = PANEL_W;   // column panel width in pixels, rounded to a multiple of 32

void conv_optimized_v4(const float* in, float* out, const float* ker,
                       int H, int W, int K) {
    if (K > kMaxK) {
        conv_optimized_v2(in, out, ker, H, W, K);
        return;
    }
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int BW = 32;
    const int H_blocked = (H / 2) * 2;

    int PW = (kPanelW / BW) * BW;
    if (PW < BW) PW = BW;
    if (PW > W) PW = W;

    __m256 kv[kMaxK * kMaxK];
    for (int i = 0; i < K * K; ++i) kv[i] = _mm256_set1_ps(ker[i]);

    for (int x0 = 0; x0 < W; x0 += PW) {                 // TILE: column panel
        const int x1 = (x0 + PW < W) ? (x0 + PW) : W;
        const int x_blocked = x0 + ((x1 - x0) / BW) * BW;

        for (int oy = 0; oy < H_blocked; oy += 2) {      // UNROLL: 2 output rows
            for (int ox = x0; ox < x_blocked; ox += BW) {  // SIMD + 4x unroll
                __m256 a0 = _mm256_setzero_ps();
                __m256 a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps();
                __m256 a3 = _mm256_setzero_ps();
                __m256 b0 = _mm256_setzero_ps();
                __m256 b1 = _mm256_setzero_ps();
                __m256 b2 = _mm256_setzero_ps();
                __m256 b3 = _mm256_setzero_ps();

                for (int r = 0; r < K + 1; ++r) {
                    const int in_offset = (oy + r) * in_stride + ox;
                    const bool useA = (r < K);
                    const bool useB = (r >= 1);

                    for (int kx = 0; kx < K; ++kx) {
                        const __m256 v0 = _mm256_loadu_ps(&in[in_offset + kx]);
                        const __m256 v1 = _mm256_loadu_ps(&in[in_offset + kx + 8]);
                        const __m256 v2 = _mm256_loadu_ps(&in[in_offset + kx + 16]);
                        const __m256 v3 = _mm256_loadu_ps(&in[in_offset + kx + 24]);

                        if (useA) {
                            const __m256 ka = kv[r * K + kx];
                            a0 = _mm256_fmadd_ps(ka, v0, a0);
                            a1 = _mm256_fmadd_ps(ka, v1, a1);
                            a2 = _mm256_fmadd_ps(ka, v2, a2);
                            a3 = _mm256_fmadd_ps(ka, v3, a3);
                        }
                        if (useB) {
                            const __m256 kb = kv[(r - 1) * K + kx];
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
            conv_opt_scalar_range(in, out, ker, oy, x_blocked, x1, W, in_stride, K);
            conv_opt_scalar_range(in, out, ker, oy + 1, x_blocked, x1, W, in_stride, K);
        }
        for (int oy = H_blocked; oy < H; ++oy) {
            conv_opt_scalar_range(in, out, ker, oy, x0, x1, W, in_stride, K);
        }
    }
}

// ===========================================================================
// v5 : v4 with the 4x2 block specialised for K = 3
// ---------------------------------------------------------------------------
// Everything above is written for general odd K, which means the compiler sees a
// runtime trip count and the useA/useB branches, and it cannot keep the eight
// accumulators in registers with full confidence. Writing the K = 3 case out by
// hand (exactly the way conv_unroll wrote its 2x2 block out by hand) removes all
// of that: the block becomes a straight-line stretch of 48 loads and 72 FMAs
// producing 64 pixels, with no branches and no loop overhead.
//
// 72 FMAs / 2 per cycle = 36 cycles for 64 pixels = 0.5625 cycles/pixel, which is
// the hardware peak for a kernel that actually evaluates all 9 taps. Anything
// beyond this would have to compute less, not compute faster.
// ===========================================================================
void conv_optimized_v5(const float* in, float* out, const float* ker,
                       int H, int W, int K) {
    if (K != 3) {                       // only K = 3 is specialised here
        conv_optimized_v4(in, out, ker, H, W, K);
        return;
    }
    const int in_stride = W + 2;        // p = 1
    const int BW = 32;
    const int H_blocked = (H / 2) * 2;

    int PW = (kPanelW / BW) * BW;
    if (PW < BW) PW = BW;
    if (PW > W) PW = W;

    __m256 k[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) k[i][j] = _mm256_set1_ps(ker[i * 3 + j]);

    for (int x0 = 0; x0 < W; x0 += PW) {
        const int x1 = (x0 + PW < W) ? (x0 + PW) : W;
        const int x_blocked = x0 + ((x1 - x0) / BW) * BW;

        for (int oy = 0; oy < H_blocked; oy += 2) {
            const float* r0 = &in[oy * in_stride];
            const float* r1 = r0 + in_stride;
            const float* r2 = r1 + in_stride;
            const float* r3 = r2 + in_stride;
            float* o0 = &out[oy * W];
            float* o1 = o0 + W;

            for (int ox = x0; ox < x_blocked; ox += BW) {
                __m256 a0 = _mm256_setzero_ps();
                __m256 a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps();
                __m256 a3 = _mm256_setzero_ps();
                __m256 b0 = _mm256_setzero_ps();
                __m256 b1 = _mm256_setzero_ps();
                __m256 b2 = _mm256_setzero_ps();
                __m256 b3 = _mm256_setzero_ps();

                // input row 0: top output row only (kernel row 0)
                for (int j = 0; j < 3; ++j) {
                    a0 = _mm256_fmadd_ps(k[0][j], _mm256_loadu_ps(r0 + ox + j), a0);
                    a1 = _mm256_fmadd_ps(k[0][j], _mm256_loadu_ps(r0 + ox + j + 8), a1);
                    a2 = _mm256_fmadd_ps(k[0][j], _mm256_loadu_ps(r0 + ox + j + 16), a2);
                    a3 = _mm256_fmadd_ps(k[0][j], _mm256_loadu_ps(r0 + ox + j + 24), a3);
                }
                // input row 1: shared -- one load feeds two FMAs
                for (int j = 0; j < 3; ++j) {
                    __m256 v;
                    v = _mm256_loadu_ps(r1 + ox + j);
                    a0 = _mm256_fmadd_ps(k[1][j], v, a0);
                    b0 = _mm256_fmadd_ps(k[0][j], v, b0);
                    v = _mm256_loadu_ps(r1 + ox + j + 8);
                    a1 = _mm256_fmadd_ps(k[1][j], v, a1);
                    b1 = _mm256_fmadd_ps(k[0][j], v, b1);
                    v = _mm256_loadu_ps(r1 + ox + j + 16);
                    a2 = _mm256_fmadd_ps(k[1][j], v, a2);
                    b2 = _mm256_fmadd_ps(k[0][j], v, b2);
                    v = _mm256_loadu_ps(r1 + ox + j + 24);
                    a3 = _mm256_fmadd_ps(k[1][j], v, a3);
                    b3 = _mm256_fmadd_ps(k[0][j], v, b3);
                }
                // input row 2: shared
                for (int j = 0; j < 3; ++j) {
                    __m256 v;
                    v = _mm256_loadu_ps(r2 + ox + j);
                    a0 = _mm256_fmadd_ps(k[2][j], v, a0);
                    b0 = _mm256_fmadd_ps(k[1][j], v, b0);
                    v = _mm256_loadu_ps(r2 + ox + j + 8);
                    a1 = _mm256_fmadd_ps(k[2][j], v, a1);
                    b1 = _mm256_fmadd_ps(k[1][j], v, b1);
                    v = _mm256_loadu_ps(r2 + ox + j + 16);
                    a2 = _mm256_fmadd_ps(k[2][j], v, a2);
                    b2 = _mm256_fmadd_ps(k[1][j], v, b2);
                    v = _mm256_loadu_ps(r2 + ox + j + 24);
                    a3 = _mm256_fmadd_ps(k[2][j], v, a3);
                    b3 = _mm256_fmadd_ps(k[1][j], v, b3);
                }
                // input row 3: bottom output row only (kernel row 2)
                for (int j = 0; j < 3; ++j) {
                    b0 = _mm256_fmadd_ps(k[2][j], _mm256_loadu_ps(r3 + ox + j), b0);
                    b1 = _mm256_fmadd_ps(k[2][j], _mm256_loadu_ps(r3 + ox + j + 8), b1);
                    b2 = _mm256_fmadd_ps(k[2][j], _mm256_loadu_ps(r3 + ox + j + 16), b2);
                    b3 = _mm256_fmadd_ps(k[2][j], _mm256_loadu_ps(r3 + ox + j + 24), b3);
                }

                _mm256_storeu_ps(o0 + ox + 0, a0);
                _mm256_storeu_ps(o0 + ox + 8, a1);
                _mm256_storeu_ps(o0 + ox + 16, a2);
                _mm256_storeu_ps(o0 + ox + 24, a3);
                _mm256_storeu_ps(o1 + ox + 0, b0);
                _mm256_storeu_ps(o1 + ox + 8, b1);
                _mm256_storeu_ps(o1 + ox + 16, b2);
                _mm256_storeu_ps(o1 + ox + 24, b3);
            }
            conv_opt_scalar_range(in, out, ker, oy, x_blocked, x1, W, in_stride, 3);
            conv_opt_scalar_range(in, out, ker, oy + 1, x_blocked, x1, W, in_stride, 3);
        }
        for (int oy = H_blocked; oy < H; ++oy) {
            conv_opt_scalar_range(in, out, ker, oy, x0, x1, W, in_stride, 3);
        }
    }
}

// ---------------------------------------------------------------------------
// Pick the winner. Swap the call to reproduce the table in the report:
//   v1 : simd + unroll(ox)                       -- baseline of this stage
//   v2 : + unroll(oy), input rows reused          -- fewer loads per pixel
//   v3 : + kernel broadcasts hoisted              -- fewer uops in the hot loop
//   v4 : + column-panel tiling                    -- shorter reuse distance
//   v5 : + K = 3 block written out by hand        -- branch-free straight line
// ---------------------------------------------------------------------------
void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    conv_optimized_v1(in, out, ker, H, W, K);
}

// This main function is only used for profiling and is done so with the same
// compiler optimizations as were there in the makefile.
// This was necessary as the main.cpp has a lot of overhead such as running naive
// This messes up the readings given by the perf command.
// The command used to run this
//g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall -DSTANDALONE_TEST src/conv_optimized.cpp -o bin/conv_prof_optimized
#ifdef STANDALONE_TEST
int main(int argc, char** argv) {
    std::printf("PROFILING OPTIMIZED.\n");
    int H = 2048, W = 2048, K = 3;
    unsigned seed = 1234;
    if (argc >= 4) {
        H = std::atoi(argv[1]);
        W = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
        std::printf("Setting H=%d, W=%d, K=%d\n", H, W, K);
    }
    if (argc >= 6) seed = static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10));
    float* img = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    float* ker = pa1::alloc_floats(static_cast<std::size_t>(K) * K);
    float* out = pa1::alloc_floats(static_cast<std::size_t>(H) * W);

    pa1::fill_random(img, static_cast<std::size_t>(H) * W, seed);
    pa1::fill_random(ker, static_cast<std::size_t>(K) * K, seed + 1u);
    float* in = pa1::make_padded(img, H, W, K);  // zero-padded halo buffer, stride W+2p
    conv_optimized(in, out, ker, H, W, K);
    return 0;
}
#endif