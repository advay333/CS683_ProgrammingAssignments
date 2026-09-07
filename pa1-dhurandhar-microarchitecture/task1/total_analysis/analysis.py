#!/usr/bin/env python3
"""
summary_figure.py -- "best of each optimisation stage" comparison.

Builds one tidy CSV and the grouped bar chart (Figure 1.1 style): normalised
speedup over the unoptimised naive baseline, grouped by matrix size, one bar
per optimisation stage.

Stage selection: for each matrix size independently, the best-performing
variant within that stage is taken. Which variant won is recorded in the CSV
so the choice is auditable rather than hidden.

  Loop reordering  only one variant (base)
  Loop unrolling   best of v4, v5                     -> v5 at every size
  Tiling           best of tile sizes 8..1024         -> varies
  SIMD             best of simd128_v2..simd256_v3     -> varies
  Optimised        the final combined implementation

Outputs (--out):
    speedup_best_per_stage.csv    tidy long form, one row per (size, stage)
    speedup_best_wide.csv         wide form, sizes x stages
    fig_stage_comparison.pdf/png  all nine matrix sizes
    fig_stage_comparison_4.pdf/png  four representative sizes (report figure)
    fig_stage_comparison_log.pdf/png  log y-axis version

Usage:
    python3 summary_figure.py --out ./out
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
TILES = [8, 16, 32, 64, 128, 256, 512, 1024]

# Sizes shown in the four-group report figure. Chosen to span the regimes:
# small/in-cache, the peak, the transition, and fully DRAM-bound.
REPORT_SIZES = [256, 1024, 4096, 16384]

# ---------------------------------------------------------------------------
# Measured speedups over the naive baseline (median of 20 runs)
# ---------------------------------------------------------------------------

REORDER = pd.DataFrame(
    {"base": [1.45, 1.45, 1.49, 1.50, 1.52, 1.02, 0.88, 0.89, 0.89]},
    index=SIZES)

UNROLL = pd.DataFrame(
    {"v4": [2.12, 2.20, 2.06, 2.06, 1.93, 1.88, 1.96, 2.10, 2.09],
     "v5": [3.04, 3.13, 2.95, 3.13, 3.03, 2.95, 2.92, 2.92, 2.79]},
    index=SIZES)

SIMD = pd.DataFrame(
    {"simd128_v2": [3.83, 4.13, 4.18, 4.33, 4.46, 4.27, 4.25, 4.19, 4.08],
     "simd128_v3": [6.70, 7.98, 8.40, 8.98, 8.84, 7.76, 7.29, 6.99, 6.78],
     "simd128_v4": [10.11, 11.32, 11.12, 11.39, 11.48, 9.44, 8.36, 8.14, 7.73],
     "simd256_v1": [5.31, 6.37, 7.88, 8.40, 8.44, 7.41, 7.02, 5.41, 5.17],
     "simd256_v2": [7.81, 9.18, 13.10, 13.37, 13.88, 10.43, 8.94, 7.68, 7.17],
     "simd256_v3": [10.25, 10.98, 13.98, 14.48, 14.52, 10.98, 9.13, 7.75, 7.16]},
    index=SIZES)

TILE = pd.DataFrame(
    [[0.86, 0.98, 0.93, 1.09, 1.00, 1.00, 0.96, 1.00],
     [0.86, 0.95, 0.90, 0.94, 1.03, 1.00, 0.97, 0.98],
     [1.33, 1.27, 1.21, 1.22, 1.30, 1.38, 1.39, 1.28],
     [0.85, 0.91, 0.91, 0.95, 0.96, 0.95, 0.97, 1.00],
     [0.83, 0.92, 0.89, 0.98, 0.99, 1.03, 0.99, 1.02],
     [0.82, 0.91, 0.91, 0.95, 0.97, 1.00, 1.00, 1.02],
     [0.84, 0.93, 0.91, 0.95, 0.96, 1.03, 0.97, 1.02],
     [0.85, 0.94, 0.90, 0.97, 0.99, 0.99, 1.02, 1.01],
     [0.84, 0.93, 0.93, 0.99, 1.03, 1.00, 1.01, 1.02]],
    index=SIZES, columns=TILES)

TILE_SD = pd.DataFrame(
    [[0.12, 0.07, 0.17, 0.10, 0.15, 0.11, 0.16, 0.14],
     [0.14, 0.10, 0.17, 0.17, 0.19, 0.16, 0.19, 0.15],
     [0.10, 0.12, 0.18, 0.11, 0.17, 0.13, 0.10, 0.12],
     [0.07, 0.10, 0.11, 0.09, 0.14, 0.10, 0.12, 0.12],
     [0.05, 0.06, 0.06, 0.04, 0.08, 0.09, 0.10, 0.08],
     [0.06, 0.05, 0.06, 0.05, 0.06, 0.05, 0.04, 0.08],
     [0.05, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.06],
     [0.05, 0.04, 0.05, 0.03, 0.04, 0.07, 0.05, 0.04],
     [0.06, 0.06, 0.06, 0.04, 0.06, 0.06, 0.07, 0.07]],
    index=SIZES, columns=TILES)

OPTIMISED = pd.Series(
    [10.72, 11.34, 14.03, 17.39, 18.10, 15.33, 14.18, 14.01, 8.11],
    index=SIZES)
OPTIMISED_SD = pd.Series(
    [0.01, 0.01, 0.01, 0.00, 0.00, 0.01, 0.01, 0.02, 0.02],
    index=SIZES)

# Display order and colours. Order follows the pipeline, not magnitude, so the
# reader sees each stage building on the last.
STAGES = ["Loop reordering", "Loop unrolling", "Tiling", "SIMD", "Optimised"]
COLORS = {
    "Loop reordering": "#4472C4",
    "Loop unrolling":  "#ED7D31",
    "Tiling":          "#A5A5A5",
    "SIMD":            "#FFC000",
    "Optimised":       "#5B9BD5",
}


def build_table():
    """Pick the winning variant per stage per size, and record which one won."""
    rows = []
    for n in SIZES:
        picks = [
            ("Loop reordering", "base", REORDER.loc[n, "base"], np.nan),
            ("Loop unrolling", UNROLL.loc[n].idxmax(), UNROLL.loc[n].max(), np.nan),
            ("Tiling", f"tile_{TILE.loc[n].idxmax()}", TILE.loc[n].max(),
             TILE_SD.loc[n, TILE.loc[n].idxmax()]),
            ("SIMD", SIMD.loc[n].idxmax(), SIMD.loc[n].max(), np.nan),
            ("Optimised", "conv_optimized", OPTIMISED[n], OPTIMISED_SD[n]),
        ]
        for stage, variant, sp, sd in picks:
            rows.append(dict(matrix_size=n, stage=stage, best_variant=variant,
                             speedup=round(float(sp), 3),
                             std_dev=(round(float(sd), 3) if not np.isnan(sd) else ""),
                             note=("baseline artifact at N=256"
                                   if (stage == "Tiling" and n == 256) else "")))
    long = pd.DataFrame(rows)
    wide = long.pivot(index="matrix_size", columns="stage",
                      values="speedup")[STAGES]
    return long, wide


def bar_chart(wide, sizes, path, logy=False, figsize=(9.2, 4.8), title=None):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(sizes))
    w = 0.16
    for i, st in enumerate(STAGES):
        vals = [wide.loc[n, st] for n in sizes]
        bars = ax.bar(x + (i - 2) * w, vals, w, label=st, color=COLORS[st],
                      edgecolor="white", linewidth=0.5, zorder=3)
        ax.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=1.5, rotation=90)

    # A speedup of 1.0 is "no change". Anything below is a regression, so mark it.
    ax.axhline(1.0, color="#c00000", lw=1.2, ls="--", zorder=4,
               label="no change ($1.0\\times$)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("Matrix size $N$ ($N\\times N$)")
    ax.set_ylabel("Normalised speedup\n(vs. no optimisation)")
    ax.set_title(title or "Best variant of each optimisation stage")
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(0.7, 40)
        ax.set_yticks([1, 2, 5, 10, 20])
        ax.set_yticklabels(["1", "2", "5", "10", "20"])
    else:
        ax.set_ylim(0, max(wide.loc[sizes].values.max() * 1.22, 2))
    ax.legend(frameon=False, ncol=6, fontsize=8.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    long, wide = build_table()
    long.to_csv(os.path.join(a.out, "speedup_best_per_stage.csv"), index=False)
    wide.round(2).to_csv(os.path.join(a.out, "speedup_best_wide.csv"))

    bar_chart(wide, SIZES, os.path.join(a.out, "fig_stage_comparison"))
    bar_chart(wide, REPORT_SIZES, os.path.join(a.out, "fig_stage_comparison_4"),
              figsize=(7.4, 4.6))
    bar_chart(wide, SIZES, os.path.join(a.out, "fig_stage_comparison_log"),
              logy=True,
              title="Best variant of each optimisation stage (log scale)")

    pd.set_option("display.width", 200)
    print(wide.round(2).to_string())
    print("\nwinning variant per stage:")
    print(long.pivot(index="matrix_size", columns="stage",
                     values="best_variant")[STAGES].to_string())
    print(f"\nwrote CSVs and figures to {a.out}")


if __name__ == "__main__":
    main()