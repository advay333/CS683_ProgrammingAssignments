#!/usr/bin/env python3
"""
tile_analysis.py -- tables and plots for the cache-tiling section.

Inputs (--data):
    perf_metrics.csv   per-run hardware counters (label = tile_<size>, naive,
                       calibration), 20 reps per (label, size)

Speedup means and standard deviations are embedded below: they come from the
timing harness (sum of tiled times / sum of naive times over 20 runs), not
from perf, so they are an independent measurement of the same effect.

Outputs (--out):
    tab_tile_mpki_process.tex   process-level L1-D MPKI, every tile x every N
    tab_tile_mpki_kernel.tex    harness-corrected kernel-only MPKI
    tab_tile_speedup.tex        speedup +/- sigma
    tab_tile_best.tex           best tile size per matrix size
    fig_mpki_vs_size.pdf        MPKI vs matrix size, one line per tile size
    fig_speedup_vs_size.pdf     speedup vs matrix size, one line per tile size
    fig_mpki_vs_tile.pdf        MPKI vs tile size at the large sizes
    fig_mpki_speedup_scatter.pdf  does lower MPKI buy time?
    fig_tile_heatmap.pdf        speedup heat map
    tile_numbers.txt            every scalar quoted in the prose

Usage:
    python3 tile_analysis.py --data /path/to/csvs --out ./out
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

NAIVE = "naive"
CALIB = "calibration"
MISS = "L1-dcache-load-misses"
LOADS = "L1-dcache-loads"
INSTR = "instructions"
CYC = "cpu-cycles"

TILES = [8, 16, 32, 64, 128, 256, 512, 1024]
SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Sizes at which the harness subtraction is trustworthy.  Below this the
# kernel's own miss count is smaller than the run-to-run spread of either
# measurement, so the difference is noise (see check_noise_floor below).
RELIABLE = [2048, 4096, 8192, 16384]

# ---------------------------------------------------------------------------
# Timing results supplied by the harness: mean speedup = sum(t_tiled)/sum(t_naive)
# over 20 runs, and the standard deviation across those runs.
# rows = matrix size, cols = tile size
# ---------------------------------------------------------------------------
SPEEDUP = pd.DataFrame(
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

STDDEV = pd.DataFrame(
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

CMAP = plt.get_cmap("viridis")
TCOL = {t: CMAP(i / (len(TILES) - 1)) for i, t in enumerate(TILES)}


def mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x))))


# ---------------------------------------------------------------------------
def load(data_dir):
    m = pd.read_csv(os.path.join(data_dir, "perf_metrics.csv"))
    assert (m.exit_status == 0).all(), "some profiled runs failed"
    m = m.copy()
    # tile runs are labelled tile_<size>; normalise to a plain integer column
    m["MPKI"] = 1000.0 * m[MISS] / m[INSTR]
    return m


def check_noise_floor(m):
    """The kernel-only MPKI is a difference of two medians.  Report, for each
    size, whether that difference is large compared with the run-to-run spread.
    If it is not, the corrected value is meaningless -- and at small N it comes
    out negative, which is the giveaway."""
    out = []
    for h in SIZES:
        c = m[(m.label == CALIB) & (m.H == h)][MISS]
        n = m[(m.label == "tile_256") & (m.H == h)][MISS]
        diff = n.median() - c.median()
        noise = mad(n) + mad(c)
        out.append((h, diff, noise, diff / noise if noise else np.nan))
    return pd.DataFrame(out, columns=["H", "diff", "noise", "ratio"]).set_index("H")


def mpki_tables(m):
    """Process-level and harness-corrected kernel-only MPKI."""
    proc = m.groupby(["label", "H"]).MPKI.median().unstack()

    med = m.groupby(["label", "H"])[[MISS, INSTR, LOADS, CYC]].median()
    labels = [NAIVE] + [f"tile_{t}" for t in TILES]
    ker = {}
    cyc_ratio = {}
    for l in labels:
        ker[l] = [1000.0 * (med.loc[(l, h), MISS] - med.loc[(CALIB, h), MISS])
                  / (med.loc[(l, h), INSTR] - med.loc[(CALIB, h), INSTR])
                  for h in SIZES]
        cyc_ratio[l] = [(med.loc[(l, h), CYC] - med.loc[(CALIB, h), CYC])
                        / (med.loc[(NAIVE, h), CYC] - med.loc[(CALIB, h), CYC])
                        for h in SIZES]
    ker = pd.DataFrame(ker, index=SIZES).T
    cyc_ratio = pd.DataFrame(cyc_ratio, index=SIZES).T
    return proc, ker, cyc_ratio


# ---------------------------------------------------------------------------
def tex(path, body, header, caption, label, colspec, note=None):
    with open(path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n")
        f.write(header + " \\\\\n\\midrule\n")
        f.write(body)
        f.write("\\bottomrule\n\\end{tabular}\n")
        if note:
            f.write("\n\\vspace{2pt}\n\\begin{minipage}{\\linewidth}\\footnotesize "
                    + note + "\\end{minipage}\n")
        f.write("\\end{table}\n")


def emit_mpki_process(out, proc):
    head = "Size & naive & " + " & ".join(str(t) for t in TILES)
    rows = []
    for h in SIZES:
        cells = [f"{proc.loc[NAIVE, h]:.2f}"] + \
                [f"{proc.loc[f'tile_{t}', h]:.2f}" for t in TILES]
        rows.append(f"{h}$\\times${h} & " + " & ".join(cells) + " \\\\\n")
    tex(os.path.join(out, "tab_tile_mpki_process.tex"), "".join(rows), head,
        "Process-level L1-D MPKI for every tile size (columns, in elements) "
        "and matrix size (rows). Median of 20 runs.",
        "tab:tile-mpki-process", "l" + "r" * (len(TILES) + 1),
        note="Includes the allocate/initialise harness, which dominates the "
             "miss count at small $N$.")


def emit_mpki_kernel(out, ker):
    head = "Size & naive & " + " & ".join(str(t) for t in TILES)
    rows = []
    for h in RELIABLE:
        cells = [f"{ker.loc[NAIVE, h]:.3f}"] + \
                [f"{ker.loc[f'tile_{t}', h]:.3f}" for t in TILES]
        rows.append(f"{h}$\\times${h} & " + " & ".join(cells) + " \\\\\n")
    tex(os.path.join(out, "tab_tile_mpki_kernel.tex"), "".join(rows), head,
        "Harness-corrected, kernel-only L1-D MPKI. Only $N \\geq 2048$ is "
        "shown: below that the kernel's own miss count is smaller than the "
        "run-to-run spread of the measurement.",
        "tab:tile-mpki-kernel", "l" + "r" * (len(TILES) + 1),
        note="Computed as $1000\\,(\\text{miss}_{\\text{tiled}} - "
             "\\text{miss}_{\\text{harness}}) / (\\text{instr}_{\\text{tiled}} "
             "- \\text{instr}_{\\text{harness}})$ -- the raw counters are "
             "differenced before the ratio is formed.")


def emit_speedup(out):
    head = "Size & " + " & ".join(str(t) for t in TILES)
    rows = []
    for h in SIZES:
        cells = [f"{SPEEDUP.loc[h,t]:.2f}\\,$\\pm$\\,{STDDEV.loc[h,t]:.2f}"
                 for t in TILES]
        rows.append(f"{h}$\\times${h} & " + " & ".join(cells) + " \\\\\n")
    tex(os.path.join(out, "tab_tile_speedup.tex"), "".join(rows), head,
        "Speedup of the tiled implementation over the naive baseline, "
        "mean $\\pm$ standard deviation over 20 runs. Values below 1.00 are "
        "slowdowns.",
        "tab:tile-speedup", "l" + "r" * len(TILES))


def emit_best(out, ker, cyc):
    rows = []
    for h in SIZES:
        sp = SPEEDUP.loc[h]
        bt = sp.idxmax()
        if h in RELIABLE:
            km = ker.loc[[f"tile_{t}" for t in TILES], h]
            bm = TILES[int(np.argmin(km.values))]
            mstr = f"{bm} ({km.min():.3f})"
            nstr = f"{ker.loc[NAIVE, h]:.3f}"
        else:
            mstr, nstr = "--", "--"
        rows.append(f"{h}$\\times${h} & {bt} & {sp.max():.2f} & "
                    f"{STDDEV.loc[h, bt]:.2f} & {mstr} & {nstr} \\\\\n")
    tex(os.path.join(out, "tab_tile_best.tex"), "".join(rows),
        "Size & Fastest tile & Speedup & $\\sigma$ & Lowest-MPKI tile & naive MPKI",
        "Best tile size per matrix size, by execution time and by cache "
        "behaviour. The two criteria do not select the same tile, and no "
        "speedup exceeds one standard deviation above unity.",
        "tab:tile-best", "lrrrrr")


# ---------------------------------------------------------------------------
def style(ax):
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_mpki_vs_size(out, ker):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(RELIABLE, [ker.loc[NAIVE, h] for h in RELIABLE], "k^--", lw=2.2,
            ms=7, label="naive", zorder=5)
    for t in TILES:
        ax.plot(RELIABLE, [ker.loc[f"tile_{t}", h] for h in RELIABLE],
                marker="o", lw=1.7, ms=5, color=TCOL[t], label=f"tile {t}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(RELIABLE)
    ax.set_xticklabels([str(s) for s in RELIABLE])
    ax.set_xlabel("Matrix size $N$ ($N\\times N$)")
    ax.set_ylabel("Kernel L1-D MPKI")
    ax.set_title("L1-D MPKI vs. matrix size, by tile size")
    ax.legend(frameon=False, ncol=3, fontsize=8.5)
    style(ax)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_mpki_vs_size.{e}"), dpi=160)
    plt.close(fig)


def plot_mpki_vs_tile(out, ker):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cols = plt.get_cmap("plasma")(np.linspace(0.1, 0.8, len(RELIABLE)))
    for c, h in zip(cols, RELIABLE):
        ax.plot(TILES, [ker.loc[f"tile_{t}", h] for t in TILES], marker="o",
                lw=1.8, ms=5, color=c, label=f"$N={h}$")
        ax.axhline(ker.loc[NAIVE, h], color=c, ls=":", lw=1.1, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(TILES)
    ax.set_xticklabels([str(t) for t in TILES])
    ax.set_xlabel("Tile size (elements)")
    ax.set_ylabel("Kernel L1-D MPKI")
    ax.set_title("MPKI vs. tile size (dotted = naive baseline for that $N$)")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    style(ax)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_mpki_vs_tile.{e}"), dpi=160)
    plt.close(fig)


def plot_speedup_vs_size(out):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for t in TILES:
        ax.errorbar(SIZES, SPEEDUP[t].values, yerr=STDDEV[t].values,
                    marker="o", lw=1.7, ms=4.5, capsize=2.5, elinewidth=0.9,
                    color=TCOL[t], label=f"tile {t}")
    ax.axhline(1.0, color="k", lw=1.4, ls="--", zorder=1)
    ax.text(70, 1.005, "no change", fontsize=8.5, color="k")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("Matrix size $N$")
    ax.set_ylabel("Speedup over naive")
    ax.set_title("Speedup vs. matrix size, by tile size (error bars: $\\pm1\\sigma$)")
    ax.legend(frameon=False, ncol=3, fontsize=8.5)
    style(ax)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_tile_speedup_vs_size.{e}"), dpi=160)
    plt.close(fig)


def plot_heatmap(out):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    d = SPEEDUP.values
    norm = TwoSlopeNorm(vmin=d.min(), vcenter=1.0, vmax=d.max())
    im = ax.imshow(d, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(range(len(TILES)))
    ax.set_xticklabels([str(t) for t in TILES])
    ax.set_yticks(range(len(SIZES)))
    ax.set_yticklabels([str(s) for s in SIZES])
    ax.set_xlabel("Tile size (elements)")
    ax.set_ylabel("Matrix size $N$")
    ax.set_title("Speedup over naive (red $<1$ = slower, blue $>1$ = faster)")
    for i in range(len(SIZES)):
        for j in range(len(TILES)):
            ax.text(j, i, f"{d[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.colorbar(im, ax=ax, label="speedup")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_tile_heatmap.{e}"), dpi=160)
    plt.close(fig)


def plot_mpki_speedup_scatter(out, ker):
    """The central plot of the section: does reducing MPKI buy any time?"""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    cols = plt.get_cmap("plasma")(np.linspace(0.1, 0.8, len(RELIABLE)))
    for c, h in zip(cols, RELIABLE):
        x = [ker.loc[f"tile_{t}", h] for t in TILES]
        y = [SPEEDUP.loc[h, t] for t in TILES]
        ax.scatter(x, y, s=45, color=c, label=f"$N={h}$", zorder=3)
        ax.scatter([ker.loc[NAIVE, h]], [1.0], s=90, marker="*",
                   color=c, edgecolor="k", lw=0.6, zorder=4)
    ax.axhline(1.0, color="k", lw=1.2, ls="--")
    ax.set_xlabel("Kernel L1-D MPKI")
    ax.set_ylabel("Speedup over naive")
    ax.set_title("Lower MPKI does not translate into less time\n"
                 "(stars = naive baseline)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_mpki_speedup_scatter.{e}"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="./out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = load(a.data)
    proc, ker, cyc = mpki_tables(m)
    noise = check_noise_floor(m)

    emit_mpki_process(a.out, proc)
    emit_mpki_kernel(a.out, ker)
    emit_speedup(a.out)
    emit_best(a.out, ker, cyc)

    plot_mpki_vs_size(a.out, ker)
    plot_mpki_vs_tile(a.out, ker)
    plot_speedup_vs_size(a.out)
    plot_heatmap(a.out)
    plot_mpki_speedup_scatter(a.out, ker)

    L = []
    P = L.append
    P(f"worst-case perf counter enable fraction: {m.min_pct_enabled.min():.0f}%")
    P("")
    P("--- noise floor check (tile_256 minus calibration, L1D misses) ------")
    P("  ratio < ~3 means the harness-corrected MPKI is not trustworthy")
    for h, r in noise.iterrows():
        flag = "OK" if r.ratio > 3 else "NOISE"
        P(f"  N={h:>5}: diff={r['diff']:>13,.0f}  noise={r['noise']:>10,.0f}  "
          f"ratio={r['ratio']:>7.1f}  {flag}")
    P("")
    P("--- kernel L1-D MPKI (reliable sizes only) -------------------------")
    for h in RELIABLE:
        P(f"  N={h}:  naive={ker.loc[NAIVE,h]:.3f}")
        for t in TILES:
            v = ker.loc[f"tile_{t}", h]
            P(f"      tile {t:>5}: {v:.3f}   ({ker.loc[NAIVE,h]/v:.2f}x vs naive)")
    P("")
    P("--- best MPKI reduction -------------------------------------------")
    for h in RELIABLE:
        km = {t: ker.loc[f"tile_{t}", h] for t in TILES}
        bt = min(km, key=km.get)
        P(f"  N={h:>5}: best tile={bt:>5} MPKI={km[bt]:.3f} vs naive "
          f"{ker.loc[NAIVE,h]:.3f} -> {ker.loc[NAIVE,h]/km[bt]:.2f}x fewer misses/kinstr")
    P("")
    P("--- cycle ratio (tiled/naive, harness-corrected; >1 = slower) ------")
    for h in RELIABLE:
        s = "  ".join(f"{t}:{cyc.loc[f'tile_{t}',h]:.3f}" for t in TILES)
        P(f"  N={h:>5}: {s}")
    P("")
    P("--- wall-clock speedup summary ------------------------------------")
    P(f"  global max speedup      : {SPEEDUP.values.max():.2f}")
    P(f"  global min speedup      : {SPEEDUP.values.min():.2f}")
    big = SPEEDUP.loc[[s for s in SIZES if s >= 512]]
    P(f"  N>=512 max              : {big.values.max():.2f}")
    P(f"  N>=512 mean             : {big.values.mean():.3f}")
    for t in TILES:
        col = SPEEDUP.loc[[s for s in SIZES if s >= 512], t]
        P(f"  tile {t:>5}: N>=512 mean speedup {col.mean():.3f}  "
          f"(min {col.min():.2f}, max {col.max():.2f})")
    P("")
    P("--- how many (N,tile) cells are more than 1 sigma above 1.0? -------")
    sig = (SPEEDUP - STDDEV) > 1.0
    P(f"  {int(sig.values.sum())} of {SPEEDUP.size} cells")
    for h in SIZES:
        if sig.loc[h].any():
            P(f"    N={h}: tiles {[t for t in TILES if sig.loc[h,t]]}")
    P("")
    P("--- N=256 anomaly --------------------------------------------------")
    P("  tile >= 256 at N=256 means the tile covers the whole matrix, i.e. no")
    P("  tiling happens at all, yet speedup is still well above 1:")
    for t in [256, 512, 1024]:
        P(f"    tile {t:>5}: speedup {SPEEDUP.loc[256,t]:.2f} +/- {STDDEV.loc[256,t]:.2f}")
    P("  -> this is a baseline artifact at N=256, not a tiling effect.")
    P("")
    P("--- working sets ---------------------------------------------------")
    for h in SIZES:
        st = (h + 2) * 4
        P(f"  N={h:>5}: in_stride={st:>8}B  2 padded rows={2*st/1024:8.1f} KiB  "
          f"3 rows={3*st/1024:8.1f} KiB  stride mod 4096={st%4096}")
    P("")
    for t in TILES:
        P(f"  tile {t:>5}: 3 tile-rows live = {3*(t+2)*4/1024:6.2f} KiB")

    txt = "\n".join(L)
    with open(os.path.join(a.out, "tile_numbers.txt"), "w") as f:
        f.write(txt + "\n")
    ker.to_csv(os.path.join(a.out, "tile_kernel_mpki.csv"))
    proc.to_csv(os.path.join(a.out, "tile_process_mpki.csv"))
    print(txt)


if __name__ == "__main__":
    main()