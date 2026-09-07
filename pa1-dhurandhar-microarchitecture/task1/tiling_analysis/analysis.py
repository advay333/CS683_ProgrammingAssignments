#!/usr/bin/env python3
"""
baseline_mpki.py -- L1 data-cache misses per kilo-instruction (MPKI) for the
naive 2D convolution baseline.

    MPKI = 1000 * L1-dcache-load-misses / instructions

Three numbers are computed per matrix size:

  process MPKI   whole binary as perf sees it (alloc + fill_random + padding
                 + kernel).  This is what a naive reading of perf_summary.csv
                 gives you.
  harness MPKI   the calibration build -- identical binary, no convolution
                 call -- so the setup cost on its own.
  kernel  MPKI   the convolution alone, obtained by differencing the two
                 counters *before* dividing:

                     1000 * (miss_naive - miss_calib)
                            / (instr_naive - instr_calib)

Differencing before the division matters.  MPKI is a ratio, so you cannot
subtract two MPKI values to remove the harness -- you have to subtract the raw
counts and form the ratio once.

Usage:
    python3 baseline_mpki.py --data /path/to/csvs --out ./out
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAIVE = "naive"
CALIB = "calibration"
MISS = "L1-dcache-load-misses"
LOADS = "L1-dcache-loads"
INSTR = "instructions"
K = 3
SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="./out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = pd.read_csv(os.path.join(a.data, "perf_metrics.csv"))
    assert (m.exit_status == 0).all(), "some profiled runs failed"
    assert set(m.K) == {K}

    # perf multiplexes 8 events over fewer physical counters, so raw counts are
    # extrapolated.  MPKI is a ratio of two multiplexed counters, so the
    # scaling largely cancels -- worth stating, not worth correcting.
    print(f"worst-case counter enable fraction: {m.min_pct_enabled.min():.0f}%\n")

    # ---- per-run MPKI, then median over the 20 reps ----------------------
    m = m.copy()
    m["MPKI"] = 1000.0 * m[MISS] / m[INSTR]
    m["miss_rate"] = m[MISS] / m[LOADS]

    per_run = (m[m.label.isin([NAIVE, CALIB])]
               .groupby(["label", "H"])
               .agg(MPKI_med=("MPKI", "median"),
                    MPKI_mad=("MPKI", mad),
                    missrate_med=("miss_rate", "median")))

    # ---- harness-corrected, kernel-only MPKI -----------------------------
    med = m.groupby(["label", "H"])[[MISS, INSTR, LOADS]].median()
    rows = []
    for h in SIZES:
        mn, ic = med.loc[(NAIVE, h)], med.loc[(CALIB, h)]
        d_miss = mn[MISS] - ic[MISS]
        d_instr = mn[INSTR] - ic[INSTR]
        d_loads = mn[LOADS] - ic[LOADS]
        rows.append(dict(
            H=h,
            process_MPKI=per_run.loc[(NAIVE, h), "MPKI_med"],
            process_MAD=per_run.loc[(NAIVE, h), "MPKI_mad"],
            harness_MPKI=per_run.loc[(CALIB, h), "MPKI_med"],
            kernel_MPKI=1000.0 * d_miss / d_instr,
            kernel_missrate=d_miss / d_loads,
            misses_M=d_miss / 1e6,
            instr_M=d_instr / 1e6,
            bytes_MiB=3 * h * h * 4 / 2**20,
        ))
    r = pd.DataFrame(rows).set_index("H")

    pd.set_option("display.width", 200)
    print(r.round(4).to_string())

    # ---- LaTeX table -----------------------------------------------------
    with open(os.path.join(a.out, "tab_baseline_mpki.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\\small\n")
        f.write("\\caption{L1 data-cache MPKI for the naive 2D convolution "
                "($3\\times3$ kernel, median of 20 runs). \\emph{Process} is the "
                "whole profiled binary; \\emph{Harness} is the same binary with "
                "the convolution removed; \\emph{Kernel} is formed by "
                "differencing the raw counters before taking the ratio.}\n")
        f.write("\\label{tab:baseline-mpki}\n")
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        f.write("Size & Working set & Process MPKI & Harness MPKI & "
                "Kernel MPKI & Kernel miss rate \\\\\n\\midrule\n")
        for h in SIZES:
            x = r.loc[h]
            f.write(f"{h}$\\times${h} & \\SI{{{x.bytes_MiB:.2f}}}{{\\mebi\\byte}} & "
                    f"{x.process_MPKI:.2f} & {x.harness_MPKI:.2f} & "
                    f"{x.kernel_MPKI:.2f} & {100*x.kernel_missrate:.2f}\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---- plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(SIZES, [r.loc[h, "process_MPKI"] for h in SIZES], "o--",
            color="#888888", lw=1.6, ms=5, label="whole process")
    ax.plot(SIZES, [r.loc[h, "harness_MPKI"] for h in SIZES], "^--",
            color="#bbbbbb", lw=1.6, ms=5, label="harness only")
    ax.plot(SIZES, [r.loc[h, "kernel_MPKI"] for h in SIZES], "s-",
            color="#08519c", lw=2.0, ms=6, label="naive kernel (corrected)")
    for cap, name in [(32, "L1"), (1024, "L2"), (32768, "LLC")]:
        n = np.sqrt(cap * 1024 / 12)          # 3 float arrays -> 12 B per pixel
        if SIZES[0] < n < SIZES[-1]:
            ax.axvline(n, color="k", lw=0.8, ls=":", alpha=0.5)
            ax.text(n * 1.03, ax.get_ylim()[1] * 0.9, name, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("Matrix size $N$ ($N\\times N$)")
    ax.set_ylabel("L1-D misses per 1000 instructions")
    ax.set_title("Baseline (naive) L1 data-cache MPKI")
    ax.legend(frameon=False)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_baseline_mpki.pdf"))
    fig.savefig(os.path.join(a.out, "fig_baseline_mpki.png"), dpi=160)

    r.to_csv(os.path.join(a.out, "baseline_mpki.csv"))
    print(f"\nwrote tab_baseline_mpki.tex, fig_baseline_mpki.pdf, "
          f"baseline_mpki.csv to {a.out}")


if __name__ == "__main__":
    main()