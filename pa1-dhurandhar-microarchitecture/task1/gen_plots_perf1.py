#!/usr/bin/env python3
"""
plot_perf_stats.py

Reads perf summary CSV, subtracts calibration baseline, and plots L1 cache 
statistics (Loads, Misses, Miss Rate, MPKI) for optimized variants vs naive.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Aesthetics
# --------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.7,
    "legend.framealpha": 0.9,
    "legend.fontsize": 10,
})

def process_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    # Identify calibration data
    calib = df[df['label'] == 'calibration'].copy()
    if calib.empty:
        raise ValueError("No 'calibration' rows found in the CSV!")
    
    # Keep only the columns we need from calibration to merge
    calib_baseline = calib[['H', 'W', 'K', 
                            'L1-dcache-loads_median', 
                            'L1-dcache-load-misses_median', 
                            'instructions_median']].copy()
    calib_baseline = calib_baseline.drop_duplicates(subset=['H', 'W', 'K'])
    
    # Get actual runs (excluding calibration)
    actuals = df[df['label'] != 'calibration'].copy()
    
    # Merge actuals with calibration baselines
    merged = actuals.merge(
        calib_baseline, 
        on=['H', 'W', 'K'], 
        suffixes=('', '_calib')
    )
    
    # Subtract baseline and clamp to 0 to handle OS noise on small matrices
    merged['net_loads'] = (merged['L1-dcache-loads_median'] - merged['L1-dcache-loads_median_calib']).clip(lower=0)
    merged['net_misses'] = (merged['L1-dcache-load-misses_median'] - merged['L1-dcache-load-misses_median_calib']).clip(lower=0)
    merged['net_insts'] = (merged['instructions_median'] - merged['instructions_median_calib']).clip(lower=1) # clamp to 1 to avoid div-by-zero
    
    # Calculate derived rates based on the NET hardware events
    merged['net_miss_rate'] = merged['net_misses'] / merged['net_loads'].replace(0, 1) # avoid div-by-zero
    merged['net_mpki'] = (merged['net_misses'] / merged['net_insts']) * 1000
    
    # Sort for plotting
    merged['area'] = merged['H'] * merged['W']
    merged = merged.sort_values('area')
    merged['size_label'] = merged['H'].astype(str) + "x" + merged['W'].astype(str)
    
    return merged

def plot_comparison(df: pd.DataFrame, variant_label: str, K: int, outdir: Path):
    """Generates a 2x2 grid of plots for a specific variant vs naive."""
    
    naive = df[(df['label'] == 'naive') & (df['K'] == K)].copy()
    variant = df[(df['label'] == variant_label) & (df['K'] == K)].copy()
    
    if naive.empty or variant.empty:
        print(f"Skipping {variant_label}: Missing data for K={K}")
        return
        
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"L1 Cache Profile: {variant_label} vs naive (K={K})", fontsize=16, y=1.02, weight='bold')
    
    x_labels = variant['size_label'].tolist()
    x = np.arange(len(x_labels))
    
    metrics = [
        ('net_loads', 'Net L1 Cache Loads', True, axs[0,0]),
        ('net_misses', 'Net L1 Cache Misses', True, axs[0,1]),
        ('net_miss_rate', 'Net L1 Miss Rate', False, axs[1,0]),
        ('net_mpki', 'Net L1 MPKI (Misses / 1000 Insts)', False, axs[1,1])
    ]
    
    for metric_col, title, use_log, ax in metrics:
        ax.plot(x, naive[metric_col], marker='o', linewidth=2.5, label='naive', color='#d62728')
        ax.plot(x, variant[metric_col], marker='s', linewidth=2.5, label=variant_label, color='#1f77b4')
        
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel("Matrix Size")
        ax.legend()
        
        if use_log:
            ax.set_yscale('log')
            ax.set_ylabel("Count (Log Scale)")
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Rate")

    plt.tight_layout()
    outfile = outdir / f"{variant_label}_vs_naive_L1_stats_K{K}.png"
    plt.savefig(outfile)
    plt.close()
    print(f"Saved {outfile}")

def main():
    parser = argparse.ArgumentParser(description="Plot deeper perf stats using calibrated medians.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to perf metrics CSV")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory for plots")
    parser.add_argument("--K", type=int, default=3, help="Kernel size to filter by")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"Error: {args.csv} not found")

    outdir = args.outdir or args.csv.parent / "perf_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    # Process and calibrate data
    df = process_data(args.csv)
    
    # Find all unique non-naive variants
    variants = df[df['label'] != 'naive']['label'].unique()
    
    # Plot each variant against naive
    for variant in variants:
        plot_comparison(df, variant, args.K, outdir)

if __name__ == "__main__":
    main()