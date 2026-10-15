#!/usr/bin/env python3
"""
plot_results.py - Task 1 plots for the report, from results_summary.csv.

Usage (from the repo root, after running artifacts/parse_results.py):
    python3 artifacts/plot_results.py                      # final = spp16
    python3 artifacts/plot_results.py --best spp14         # another version
    python3 artifacts/plot_results.py --csv results_summary.csv --out figures

Writes four PNGs into figures/ (the names the LaTeX report expects):
    task1_speedup.png     speedup over the no-prefetch baseline
    task1_l1d_mpki.png    L1D load MPKI
    task1_accuracy.png    prefetch accuracy (useful / fetched from L2)
    task1_coverage.png    L1D load-miss coverage

Each plot: one group per trace + a mean group (geometric mean for speedup,
arithmetic mean for the others), comparing IP-stride with the final SPP
(the baseline is added to the MPKI plot, where it is meaningful).
"""

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# fixed colour per configuration (validated categorical slots, light mode);
# hatching keeps the bars distinguishable when printed in black and white
STYLE = {
    "baseline":  {"color": "#9a9a96", "hatch": "",     "name": "Baseline (no prefetch)"},
    "ip_stride": {"color": "#2a78d6", "hatch": "",     "name": "IP-stride"},
    "best":      {"color": "#eb6834", "hatch": "///",  "name": "SPP (final)"},
}
INK = "#2b2b29"        # text / axes
MUTED = "#8a8a85"      # grid, reference lines


def load(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    data = {}
    for r in rows:
        if not r["status"].startswith("ok"):
            continue
        data[(int(r["trace"]), r["label"])] = r
    return data


def val(data, trace, label, key):
    r = data.get((trace, label))
    if r is None or r.get(key, "") == "":
        return None
    return float(r[key])


def geomean(xs):
    xs = [x for x in xs if x is not None]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def amean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def grouped_bars(ax, groups, series, fmt):
    """groups: list of group names; series: list of (style_key, label, [values])."""
    n = len(series)
    width = 0.8 / n
    for i, (skey, label, values) in enumerate(series):
        st = STYLE[skey]
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        ys = [0 if v is None else v for v in values]
        bars = ax.bar(xs, ys, width * 0.92, color=st["color"], hatch=st["hatch"],
                      edgecolor="white", linewidth=0.8, label=label, zorder=3)
        for b, v in zip(bars, values):
            if v is None:
                continue
            ax.annotate(fmt(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)


def style_axes(ax, title, ylabel):
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.margins(y=0.15)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.getcwd()))


def main():
    ap = argparse.ArgumentParser(description="Task 1 plots for the report.")
    ap.add_argument("--best", default="spp16", help="label of the final SPP version")
    ap.add_argument("--ref", default="ip_stride", help="reference prefetcher label")
    ap.add_argument("--csv", default=os.path.join(REPO, "results_summary.csv"))
    ap.add_argument("--out", default=os.path.join(REPO, "figures"))
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"{args.csv} not found - run artifacts/parse_results.py first")
    data = load(args.csv)
    traces = sorted({t for (t, _) in data})
    for lab in ("baseline", args.ref, args.best):
        missing = [t for t in traces if (t, lab) not in data]
        if missing:
            sys.exit(f"label '{lab}' has no ok result for trace(s) {missing}")
    os.makedirs(args.out, exist_ok=True)
    STYLE["best"]["name"] = f"SPP (final, {args.best})"

    groups = [f"Trace {t}" for t in traces]

    # ---------- speedup (bars = gain over baseline, labels = speedup) ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    series = []
    for skey, lab in (("ip_stride", args.ref), ("best", args.best)):
        vals = [val(data, t, lab, "speedup") for t in traces]
        vals.append(geomean(vals))
        series.append((skey, STYLE[skey]["name"], [(v - 1) * 100 for v in vals]))
    grouped_bars(ax, groups + ["Geomean"], series, lambda v: f"{1 + v / 100:.4f}")
    style_axes(ax, "Speedup over the no-prefetch baseline", "IPC gain over baseline (%)")
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=2)
    save(fig, args.out, "task1_speedup.png")

    # ---------- L1D load MPKI ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    series = []
    for skey, lab in (("baseline", "baseline"), ("ip_stride", args.ref), ("best", args.best)):
        vals = [val(data, t, lab, "l1d_load_mpki") for t in traces]
        vals.append(amean(vals))
        series.append((skey, STYLE[skey]["name"], vals))
    grouped_bars(ax, groups + ["Mean"], series, lambda v: f"{v:.2f}")
    style_axes(ax, "L1D load MPKI (lower is better)", "Load misses per 1000 instructions")
    save(fig, args.out, "task1_l1d_mpki.png")

    # ---------- accuracy and coverage ----------
    for key, fname, title, ylabel in (
            ("accuracy", "task1_accuracy.png", "Prefetch accuracy",
             "Useful / prefetches fetched from L2 (%)"),
            ("coverage", "task1_coverage.png", "L1D load-miss coverage",
             "Baseline load misses removed (%)")):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        series = []
        for skey, lab in (("ip_stride", args.ref), ("best", args.best)):
            vals = [val(data, t, lab, key) for t in traces]
            vals = [None if v is None else v * 100 for v in vals]
            vals.append(amean(vals))
            series.append((skey, STYLE[skey]["name"], vals))
        grouped_bars(ax, groups + ["Mean"], series, lambda v: f"{v:.1f}%")
        style_axes(ax, title, ylabel)
        save(fig, args.out, fname)


if __name__ == "__main__":
    main()
