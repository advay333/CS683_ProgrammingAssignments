#!/usr/bin/env python3
"""
simd_analysis.py -- reproducible analysis for the SIMD section of the report.

Inputs (same directory or --data):
    perf_metrics.csv   per-run hardware counters (one row per run)
    perf_derived.csv   same + derived ratios (IPC, MPKI, ...)
    perf_summary.csv   pre-aggregated medians/MADs (used only for cross-checking)
    summary.csv        per-run wall-clock timings (naive_ms, stage_ms, gflops, speedup)

Outputs (into --out, default ./out):
    tab_baseline.tex        naive instruction counts
    tab_instr.tex           naive vs SIMD instruction counts + reduction factor
    tab_speedup.tex         median speedup vs naive, all variants x all sizes
    tab_gflops.tex          achieved GFLOP/s
    tab_micro.tex           IPC / L1 / branch micro-architecture table
    fig_speedup_vs_size.pdf
    fig_speedup_bars.pdf
    fig_instr_reduction.pdf
    fig_gflops.pdf
    fig_ipc.pdf
    numbers.txt             every scalar quoted in the prose

Usage:
    python3 simd_analysis.py --data /path/to/csvs --out ./out
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

K = 3                       # kernel is 3x3 for every experiment in this data set
REPS_EXPECTED = 20

# perf-side labels
NAIVE = "naive"
CALIB = "calibration"       # harness-only run: alloc + init + verify, no kernel
SIMD_LABELS = ["simd128_v2", "simd128_v3", "simd128_v4",
               "simd256_v1", "simd256_v2", "simd256_v3"]

# summary.csv uses slightly different label spellings
SUMMARY_SIMD = SIMD_LABELS
PRETTY = {
    "simd128_v2": "SSE-128 v2",
    "simd128_v3": "SSE-128 v3",
    "simd128_v4": "SSE-128 v4",
    "simd256_v1": "AVX2-256 v1",
    "simd256_v2": "AVX2-256 v2",
    "simd256_v3": "AVX2-256 v3",
    "naive":      "naive",
    "reorder_base": "loop-reordered",
    "base":       "loop-reordered",
}
COLORS = {
    "simd128_v2": "#9ecae1", "simd128_v3": "#4292c6", "simd128_v4": "#08519c",
    "simd256_v1": "#fdae6b", "simd256_v2": "#e6550d", "simd256_v3": "#a63603",
}
SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


# --------------------------------------------------------------------------
# Robust statistics
# --------------------------------------------------------------------------

def med(x):
    return float(np.median(x))


def mad(x):
    """Median absolute deviation -- robust spread, insensitive to the one or two
    runs that get descheduled or hit by a page-fault storm."""
    x = np.asarray(x, dtype=float)
    return float(np.median(np.abs(x - np.median(x))))


def agg(df, by, cols):
    """Median + MAD of `cols`, grouped by `by`."""
    g = df.groupby(by)
    out = g[cols].median()
    out.columns = [f"{c}_med" for c in cols]
    for c in cols:
        out[f"{c}_mad"] = g[c].apply(mad)
    out["n"] = g.size()
    return out.reset_index()


# --------------------------------------------------------------------------
# Load + validate
# --------------------------------------------------------------------------

def load(data_dir):
    metrics = pd.read_csv(os.path.join(data_dir, "perf_metrics.csv"))
    derived = pd.read_csv(os.path.join(data_dir, "perf_derived.csv"))
    timing = pd.read_csv(os.path.join(data_dir, "summary.csv"))

    # sanity checks -- if any of these fire, the tables below are not trustworthy
    assert (metrics.exit_status == 0).all(), "some profiled runs did not exit cleanly"
    assert (timing.correct == "yes").all(), "some runs failed the correctness check"
    assert (metrics.H == metrics.W).all() and (timing.H == timing.W).all()
    assert set(metrics.K) == {K} and set(timing.K) == {K}
    n = metrics.groupby(["label", "H"]).size()
    assert (n == REPS_EXPECTED).all(), f"expected {REPS_EXPECTED} reps/config"

    # perf multiplexes 8 events onto fewer physical counters, so every count is
    # scaled by 100/pct_enabled.  Record the worst case so it can be quoted.
    mux = float(metrics.min_pct_enabled.min())
    return metrics, derived, timing, mux


# --------------------------------------------------------------------------
# Instruction-count analysis
# --------------------------------------------------------------------------

def macs(h):
    """Multiply-accumulates in one 3x3 convolution over an h x h image."""
    return h * h * K * K


def instruction_tables(metrics):
    """perf counts the whole process, which includes a large allocate/initialise/
    verify harness whose cost also grows with H*W.  The `calibration` stage is
    exactly that harness with the kernel removed, so subtracting its median
    gives a kernel-only estimate.  Both numbers are reported."""
    piv = metrics.groupby(["label", "H"]).instructions.median().unstack()
    harness = piv.loc[CALIB]

    kernel = piv.sub(harness, axis=1)          # harness-corrected
    per_mac = kernel.div(pd.Series({h: macs(h) for h in piv.columns}), axis=1)

    reduction_raw = piv.loc[NAIVE] / piv        # raw process-level ratio
    reduction_ker = kernel.loc[NAIVE] / kernel  # kernel-only ratio
    return piv, harness, kernel, per_mac, reduction_raw, reduction_ker


# --------------------------------------------------------------------------
# LaTeX emitters
# --------------------------------------------------------------------------

def fmt(x, nd=2):
    if pd.isna(x):
        return "--"
    return f"{x:.{nd}f}"


def tex_table(path, body, header, caption, label, colspec, note=None):
    with open(path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n")
        f.write(header + " \\\\\n\\midrule\n")
        f.write(body)
        f.write("\\bottomrule\n\\end{tabular}\n")
        if note:
            f.write(f"\n\\vspace{{2pt}}\n\\begin{{minipage}}{{\\linewidth}}\\footnotesize {note}\\end{{minipage}}\n")
        f.write("\\end{table}\n")


def emit_baseline(out, piv, harness, kernel, per_mac):
    rows = []
    for h in SIZES:
        rows.append(
            f"{h}$\\times${h} & {macs(h)/1e6:,.3f} & {piv.loc[NAIVE, h]/1e6:,.1f} & "
            f"{harness[h]/1e6:,.1f} & {kernel.loc[NAIVE, h]/1e6:,.1f} & "
            f"{fmt(per_mac.loc[NAIVE, h])} \\\\\n")
    tex_table(
        os.path.join(out, "tab_baseline.tex"), "".join(rows),
        "Size & MACs (M) & Process (M) & Harness (M) & Kernel (M) & Instr./MAC",
        "Baseline profiling of the naive 2D convolution ($3\\times3$ kernel, "
        "median of 20 runs). \\emph{Process} is the raw \\texttt{perf} instruction "
        "count for the whole binary; \\emph{Harness} is the same binary with the "
        "convolution removed; \\emph{Kernel} is the difference.",
        "tab:simd-baseline", "lrrrrr",
        note="All counts in millions of retired instructions.")


def emit_instr(out, piv, kernel, reduction_ker):
    labels = [NAIVE] + SIMD_LABELS
    head = "Size & " + " & ".join(PRETTY[l] for l in labels)
    rows = []
    for h in SIZES:
        cells = [f"{kernel.loc[l, h]/1e6:,.1f}" for l in labels]
        rows.append(f"{h}$\\times${h} & " + " & ".join(cells) + " \\\\\n")
    rows.append("\\midrule\n")
    for h in SIZES:
        cells = [f"{reduction_ker.loc[l, h]:.2f}" for l in labels]
        rows.append(f"{h}$\\times${h} & " + " & ".join(cells) + " \\\\\n")
    tex_table(
        os.path.join(out, "tab_instr.tex"), "".join(rows), head,
        "Kernel-only retired instructions in millions (upper block) and "
        "instruction-count reduction factor relative to naive (lower block).",
        "tab:simd-instr", "l" + "r" * len(labels))


def emit_speedup(out, timing):
    sp = timing.groupby(["label", "H"]).speedup.median().unstack()
    spmad = timing.groupby(["label", "H"]).speedup.apply(mad).unstack()
    head = "Size & " + " & ".join(PRETTY[l] for l in SUMMARY_SIMD)
    rows = []
    for h in SIZES:
        cells = [f"{sp.loc[l, h]:.2f}" for l in SUMMARY_SIMD]
        rows.append(f"{h}$\\times${h} & " + " & ".join(cells) + " \\\\\n")
    tex_table(
        os.path.join(out, "tab_speedup.tex"), "".join(rows), head,
        "Median wall-clock speedup over the naive implementation "
        "(20 repetitions per configuration).",
        "tab:simd-speedup", "l" + "r" * len(SUMMARY_SIMD))
    return sp, spmad


def emit_gflops(out, timing):
    gf = timing.groupby(["label", "H"]).gflops.median().unstack()
    head = "Size & " + " & ".join(PRETTY[l] for l in SUMMARY_SIMD)
    rows = []
    for h in SIZES:
        cells = [f"{gf.loc[l, h]:.1f}" for l in SUMMARY_SIMD]
        rows.append(f"{h}$\\times${h} & " + " & ".join(cells) + " \\\\\n")
    tex_table(
        os.path.join(out, "tab_gflops.tex"), "".join(rows), head,
        "Achieved throughput in GFLOP/s (median of 20 runs).",
        "tab:simd-gflops", "l" + "r" * len(SUMMARY_SIMD))
    return gf


def emit_micro(out, derived):
    labels = [NAIVE, "simd128_v4", "simd256_v3"]
    ipc = derived.groupby(["label", "H"]).IPC.median().unstack()
    l1 = derived.groupby(["label", "H"]).L1D_MPKI.median().unstack()
    br = derived.groupby(["label", "H"]).branch_MPKI.median().unstack()
    rows = []
    for h in SIZES:
        cells = []
        for l in labels:
            cells += [f"{ipc.loc[l, h]:.2f}", f"{l1.loc[l, h]:.2f}", f"{br.loc[l, h]:.2f}"]
        rows.append(f"{h}$\\times${h} & " + " & ".join(cells) + " \\\\\n")
    head = ("Size & \\multicolumn{3}{c}{naive} & \\multicolumn{3}{c}{SSE-128 v4} & "
            "\\multicolumn{3}{c}{AVX2-256 v3} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
            " & IPC & L1D & Br. & IPC & L1D & Br. & IPC & L1D & Br.")
    tex_table(
        os.path.join(out, "tab_micro.tex"), "".join(rows), head,
        "Micro-architectural behaviour. L1D and Br. are L1 data-cache misses and "
        "branch mispredictions per thousand instructions (MPKI).",
        "tab:simd-micro", "l" + "rrr" * 3)
    return ipc, l1, br


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def style(ax):
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_speedup_vs_size(out, sp):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for l in SUMMARY_SIMD:
        ax.plot(SIZES, [sp.loc[l, h] for h in SIZES], "o-", lw=1.8, ms=5,
                color=COLORS[l], label=PRETTY[l])
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.axhline(4.0, color="#4292c6", lw=1, ls=":", alpha=0.8)
    ax.axhline(8.0, color="#e6550d", lw=1, ls=":", alpha=0.8)
    ax.text(70, 4.15, "SSE ideal ($4\\times$)", fontsize=8, color="#4292c6")
    ax.text(70, 8.15, "AVX2 ideal ($8\\times$)", fontsize=8, color="#e6550d")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("Matrix size $N$ ($N\\times N$)")
    ax.set_ylabel("Speedup over naive")
    ax.set_title("SIMD speedup vs. matrix size ($3\\times3$ kernel)")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_speedup_vs_size.pdf"))
    fig.savefig(os.path.join(out, "fig_speedup_vs_size.png"), dpi=160)
    plt.close(fig)


def plot_speedup_bars(out, sp):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(SIZES))
    w = 0.13
    for i, l in enumerate(SUMMARY_SIMD):
        ax.bar(x + (i - 2.5) * w, [sp.loc[l, h] for h in SIZES], w,
               color=COLORS[l], label=PRETTY[l], edgecolor="white", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SIZES], rotation=45)
    ax.set_xlabel("Matrix size $N$")
    ax.set_ylabel("Speedup over naive")
    ax.set_title("Speedup by SIMD width and matrix size")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_speedup_bars.pdf"))
    fig.savefig(os.path.join(out, "fig_speedup_bars.png"), dpi=160)
    plt.close(fig)


def plot_instr_reduction(out, reduction_ker):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for l in SIMD_LABELS:
        ax.plot(SIZES, [reduction_ker.loc[l, h] for h in SIZES], "s-", lw=1.8,
                ms=5, color=COLORS[l], label=PRETTY[l])
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("Matrix size $N$")
    ax.set_ylabel("Instruction-count reduction vs. naive ($\\times$)")
    ax.set_title("Retired-instruction reduction (kernel only)")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_instr_reduction.pdf"))
    fig.savefig(os.path.join(out, "fig_instr_reduction.png"), dpi=160)
    plt.close(fig)


def plot_gflops(out, gf, timing):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    naive_gf = (timing.groupby("H")
                .apply(lambda d: 2 * macs(d.name) / (d.naive_ms.median() * 1e6),
                       include_groups=False))
    ax.plot(SIZES, [naive_gf[h] for h in SIZES], "k^--", lw=1.6, ms=5, label="naive")
    for l in SUMMARY_SIMD:
        ax.plot(SIZES, [gf.loc[l, h] for h in SIZES], "o-", lw=1.8, ms=5,
                color=COLORS[l], label=PRETTY[l])
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("Matrix size $N$")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Achieved throughput")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_gflops.pdf"))
    fig.savefig(os.path.join(out, "fig_gflops.png"), dpi=160)
    plt.close(fig)


def plot_ipc(out, ipc):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for l in [NAIVE] + SIMD_LABELS:
        c = COLORS.get(l, "k")
        ax.plot(SIZES, [ipc.loc[l, h] for h in SIZES], marker="o", lw=1.8, ms=4,
                color=c, ls="--" if l == NAIVE else "-", label=PRETTY[l])
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("Matrix size $N$")
    ax.set_ylabel("IPC")
    ax.set_title("Instructions per cycle: fewer, denser instructions")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ipc.pdf"))
    fig.savefig(os.path.join(out, "fig_ipc.png"), dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="./out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    metrics, derived, timing, mux = load(a.data)
    piv, harness, kernel, per_mac, red_raw, red_ker = instruction_tables(metrics)

    emit_baseline(a.out, piv, harness, kernel, per_mac)
    emit_instr(a.out, piv, kernel, red_ker)
    sp, spmad = emit_speedup(a.out, timing)
    gf = emit_gflops(a.out, timing)
    ipc, l1, br = emit_micro(a.out, derived)

    plot_speedup_vs_size(a.out, sp)
    plot_speedup_bars(a.out, sp)
    plot_instr_reduction(a.out, red_ker)
    plot_gflops(a.out, gf, timing)
    plot_ipc(a.out, ipc)

    # ---- every scalar quoted in the prose -------------------------------
    L = []
    P = L.append
    P(f"worst-case perf counter multiplexing (min pct_enabled) : {mux:.0f}%")
    P("")
    P("--- naive baseline -------------------------------------------------")
    for h in SIZES:
        P(f"  N={h:>5}: process={piv.loc[NAIVE,h]/1e6:>10.1f}M  harness={harness[h]/1e6:>10.1f}M "
          f" kernel={kernel.loc[NAIVE,h]/1e6:>10.1f}M  instr/MAC={per_mac.loc[NAIVE,h]:.2f}")
    P(f"  mean instr/MAC over N>=512 : {per_mac.loc[NAIVE,[s for s in SIZES if s>=512]].mean():.2f}")
    P("")
    P("--- instructions per MAC, kernel only ------------------------------")
    for l in [NAIVE] + SIMD_LABELS:
        vals = [per_mac.loc[l, h] for h in SIZES if h >= 512]
        P(f"  {l:<12} large-N mean instr/MAC = {np.mean(vals):.3f}")
    P("")
    P("--- instruction reduction vs naive (kernel only) -------------------")
    for l in SIMD_LABELS:
        v = [red_ker.loc[l, h] for h in SIZES if h >= 512]
        P(f"  {l:<12} min={red_ker.loc[l,SIZES].min():.2f} max={red_ker.loc[l,SIZES].max():.2f} "
          f"large-N mean={np.mean(v):.2f}")
    P("")
    P("--- reduction using RAW process counts (uncorrected) ---------------")
    for l in SIMD_LABELS:
        P(f"  {l:<12} at N=16384 raw ratio = {red_raw.loc[l,16384]:.2f}")
    P("")
    P("--- wall-clock speedup --------------------------------------------")
    for l in SUMMARY_SIMD:
        b = sp.loc[l].idxmax()
        P(f"  {l:<12} peak {sp.loc[l,b]:.2f}x @ N={b} (MAD {spmad.loc[l,b]:.2f}); "
          f"N=16384 {sp.loc[l,16384]:.2f}x; N=64 {sp.loc[l,64]:.2f}x")
    P("")
    P("--- 256 vs 128 (best of each) --------------------------------------")
    for h in SIZES:
        r = sp.loc['simd256_v3', h] / sp.loc['simd128_v4', h]
        P(f"  N={h:>5}: v3_256/v4_128 = {r:.2f}x")
    P("")
    P("--- GFLOP/s --------------------------------------------------------")
    naive_gf = {h: 2 * macs(h) / (timing[timing.H == h].naive_ms.median() * 1e6) for h in SIZES}
    for h in SIZES:
        P(f"  N={h:>5}: naive={naive_gf[h]:6.2f}  best="
          f"{gf[h].max():6.2f} ({gf[h].idxmax()})")
    P("")
    P("--- IPC ------------------------------------------------------------")
    for l in [NAIVE] + SIMD_LABELS:
        P(f"  {l:<12} N=1024 IPC={ipc.loc[l,1024]:.2f}  N=16384 IPC={ipc.loc[l,16384]:.2f}")
    P("")
    P("--- L1D MPKI / branch MPKI at N=16384 ------------------------------")
    for l in [NAIVE] + SIMD_LABELS:
        P(f"  {l:<12} L1D_MPKI={l1.loc[l,16384]:.2f}  branch_MPKI={br.loc[l,16384]:.2f}")
    P("")
    P("--- working set (float32, 3 arrays) --------------------------------")
    for h in SIZES:
        P(f"  N={h:>5}: {3*h*h*4/2**20:9.2f} MiB")

    txt = "\n".join(L)
    with open(os.path.join(a.out, "numbers.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()