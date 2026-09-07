#!/usr/bin/env python3
"""
plot_gflops.py  --  CS683 PA-1 Task 1 GFLOPS plots

Reads the `summary.csv` produced by scripts/run_speedup.sh and produces 
GFLOPS-vs-matrix-size line plots. It automatically calculates the Naive GFLOPS 
from the provided speedup and plots it as a baseline on every chart.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# --------------------------------------------------------------------------
# aesthetics — a clean, print-friendly look shared by every figure
# --------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (7.6, 5.2),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12.5,
    "axes.labelweight": "medium",
    "axes.edgecolor": "#3a3a3a",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.7,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.fontsize": 11,
    "legend.edgecolor": "#cfcfcf",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


# --------------------------------------------------------------------------
# data loading / prep
# --------------------------------------------------------------------------
def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"stage", "label", "version", "H", "W", "K", "seed", "rep",
                "correct", "naive_ms", "stage_ms", "gflops", "speedup"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    n_bad = int((df["correct"] != "yes").sum())
    if n_bad:
        print(f"warning: {n_bad} row(s) in {csv_path} report INCORRECT output "
              f"and are still included in the statistics")

    df["size_label"] = df["H"].astype(str) + "x" + df["W"].astype(str)
    df["area"] = df["H"].astype(int) * df["W"].astype(int)
    
    # Calculate naive GFLOPS (since GFLOPS is inversely proportional to time)
    df["naive_gflops"] = df["gflops"] / df["speedup"]
    
    return df


def stats_for(df: pd.DataFrame, stage: str, label: str,
              K: Optional[int] = None) -> pd.DataFrame:
    """Median & IQR of GFLOPS per matrix size for one (stage, label[, K])."""
    sel = (df["stage"] == stage) & (df["label"] == label)
    if K is not None:
        sel &= (df["K"] == K)
    sub = df[sel]
    if sub.empty:
        print(f"warning: no rows found for stage='{stage}' label='{label}'"
              + (f" K={K}" if K is not None else ""))
        return sub

    # Calculate Median, 25th percentile, and 75th percentile for both optimized and naive
    g = sub.groupby(["size_label", "area"], as_index=False).agg(
        med_gflops=("gflops", "median"),
        q25_gflops=("gflops", lambda x: x.quantile(0.25)),
        q75_gflops=("gflops", lambda x: x.quantile(0.75)),
        med_naive=("naive_gflops", "median"),
        q25_naive=("naive_gflops", lambda x: x.quantile(0.25)),
        q75_naive=("naive_gflops", lambda x: x.quantile(0.75))
    )
    return g.sort_values("area").reset_index(drop=True)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def _plot_one_series(ax, g: pd.DataFrame, y_med: str, y_q25: str, y_q75: str, 
                     name: str, color: str, marker: str, linestyle: str = "-"):
    x = range(len(g))
    
    # Calculate asymmetric error bars relative to the median
    yerr_lower = g[y_med] - g[y_q25]
    yerr_upper = g[y_q75] - g[y_med]
    yerr = [yerr_lower, yerr_upper]
    
    ax.errorbar(
        x, g[y_med], yerr=yerr,
        label=name, color=color, marker=marker, markersize=7,
        linewidth=2.2, linestyle=linestyle, capsize=4, capthick=1.3, elinewidth=1.3,
        markeredgecolor="white", markeredgewidth=0.8, zorder=3,
    )
    return g["size_label"].tolist()


def _finish_axes(ax, xticklabels: List[str], title: str):
    ax.set_xticks(range(len(xticklabels)))
    
    # Keep the rotated labels
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_xlabel("Matrix size (H x W)")
    ax.set_ylabel("Median Performance (GFLOPS)")
    ax.set_title(title, pad=12)
    ax.legend(loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
    ax.set_ylim(bottom=0) # GFLOPS should start at 0


def make_plot(df: pd.DataFrame,
              series_specs: List[Tuple[str, str, str]],
              title: str, outfile: Path, K: Optional[int]):
    """series_specs: list of (stage, label, legend_name)."""
    fig, ax = plt.subplots()
    xticklabels: Optional[List[str]] = None
    plotted = False
    naive_g: Optional[pd.DataFrame] = None

    for i, (stage, label, legend_name) in enumerate(series_specs):
        g = stats_for(df, stage, label, K=K)
        if g.empty:
            continue
        labels = _plot_one_series(ax, g, "med_gflops", "q25_gflops", "q75_gflops", 
                                  legend_name, PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)])
        
        if xticklabels is None or len(labels) > len(xticklabels):
            xticklabels = labels
            
        # Store naive data to plot once at the end
        if naive_g is None:
            naive_g = g
            
        plotted = True

    if not plotted:
        plt.close(fig)
        print(f"skipped '{outfile.name}': no data found")
        return

    # Add the Naive baseline as a dashed line at the end
    if naive_g is not None:
        _plot_one_series(ax, naive_g, "med_naive", "q25_naive", "q75_naive", 
                         "Naive (Baseline)", "#404040", "x", linestyle="--")

    _finish_axes(ax, xticklabels, title)
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    print(f"wrote {outfile}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("try2/summary.csv"),
                     help="path to summary.csv produced by run_speedup.sh")
    ap.add_argument("--outdir", type=Path, default=None,
                     help="directory to write PNGs into (default: <csv dir>/plots)")
    ap.add_argument("--K", type=int, default=None,
                     help="kernel size to plot, if summary.csv has more than one "
                          "(default: the smallest K present)")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"error: {args.csv} not found")

    df = load_summary(args.csv)
    outdir = args.outdir or (args.csv.parent / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    K = args.K
    if K is None:
        ks = sorted(df["K"].unique())
        if len(ks) > 1:
            print(f"note: summary.csv has multiple kernel sizes {ks}; "
                  f"plotting K={ks[0]} (pass --K to choose another)")
        K = int(ks[0])

    # 1. reorder ------------------------------------------------------------
    make_plot(
        df, [("reorder", "base", "reorder (base)")],
        f"Reorder GFLOPS vs matrix size (K={K})",
        outdir / "reorder_gflops.png", K,
    )

    # 2. unroll v4 vs v5 ------------------------------------------------------
    make_plot(
        df, [("unroll", "v4", "unroll v4"),
             ("unroll", "v5", "unroll v5")],
        f"Unroll GFLOPS vs matrix size (K={K})",
        outdir / "unroll_gflops.png", K,
    )

    # 3. simd128_v2 vs simd256_v1 ---------------------------------------------
    make_plot(
        df, [("simd", "simd128_v2", "SIMD128 v2"),
             ("simd", "simd256_v1", "SIMD256 v1")],
        f"SIMD128 vs SIMD256 GFLOPS vs matrix size (K={K})",
        outdir / "simd128_vs_simd256_gflops.png", K,
    )

    # 4. simd128 versions -------------------------------------------------------
    make_plot(
        df, [("simd", "simd128_v2", "SIMD128 v2"),
             ("simd", "simd128_v3", "SIMD128 v3"),
             ("simd", "simd128_v4", "SIMD128 v4")],
        f"SIMD128 versions: GFLOPS vs matrix size (K={K})",
        outdir / "simd128_versions_gflops.png", K,
    )

    # 5. simd256 versions -------------------------------------------------------
    make_plot(
        df, [("simd", "simd256_v1", "SIMD256 v1"),
             ("simd", "simd256_v2", "SIMD256 v2"),
             ("simd", "simd256_v3", "SIMD256 v3")],
        f"SIMD256 versions: GFLOPS vs matrix size (K={K})",
        outdir / "simd256_versions_gflops.png", K,
    )


if __name__ == "__main__":
    main()