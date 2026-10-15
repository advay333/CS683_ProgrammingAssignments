#!/usr/bin/env python3
"""Tables and plots for the Task 2B (dynamic I-RRIP) parameter sweep.

Reads
  replacement_results/{lru,rrip,irrip}_<trace>.txt              static policies
  replacement_results/dyn_rrip_stats/dyn_rrip_stats_t<th>_l<n>_s<s>_<trace>.txt
writes into replacement_results/dyn_plots/
  dyn_sweep_all.csv / .md      every configuration and trace
  dyn_sweep_geomean.csv / .md  per configuration: geomean speedup, mean MPKIs, % I-RRIP decisions
  final_config.csv / .md       best configuration vs LRU, SRRIP, I-RRIP
  task2b_*.png                 plots (also copied to report/figures/)
"""

import csv
import glob
import math
import os
import re
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(HERE, "dyn_rrip_stats")
OUT_DIR = os.path.join(HERE, "dyn_plots")
REPORT_FIG_DIR = os.path.join(HERE, "..", "report", "figures")

TRACES = ["Trace1", "Trace2", "Trace3"]
THRESHOLDS = [0.15, 0.25, 0.45, 0.65, 0.75]
LEADERS = [16, 32, 64]
SCALES = [0.5, 1.0, 2.0]
STATIC = [("lru", "LRU"), ("rrip", "SRRIP"), ("irrip", "I-RRIP")]

PATTERNS = {
    "ipc": re.compile(r"^CPU 0 cumulative IPC:\s*([\d.]+)", re.M),
    "load_mpki": re.compile(r"^L2C LOAD\s+ACCESS:.*MPKI:\s*([\d.]+)", re.M),
    "instr_mpki": re.compile(r"^L2C INSTRUCTION LOAD MPKI:\s*([\d.]+)", re.M),
    "data_mpki": re.compile(r"^L2C DATA LOAD MPKI:\s*([\d.]+)", re.M),
}
DECISIONS = re.compile(r"^L2C_DYN DECISIONS total: (\d+) chose_IRRIP: (\d+) \(([\d.]+)%\) mode_switches: (\d+)", re.M)
FRAC = re.compile(r"^L2C_DYN FRAC mean: ([\d.]+) min: ([\d.]+) max: ([\d.]+)", re.M)
GROUP = re.compile(r"^L2C_DYN GROUPS (\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)", re.M)
NAME = re.compile(r"dyn_rrip_stats_t([\d.]+)_l(\d+)_s([\d.]+)_(Trace\d)\.txt$")


def parse(path):
    with open(path, errors="replace") as f:
        text = f.read()
    stats = {}
    for key, pattern in PATTERNS.items():
        match = pattern.search(text)
        if not match:
            sys.exit(f"'{key}' missing in {path}")
        stats[key] = float(match.group(1))
    match = DECISIONS.search(text)
    if match:
        stats["decisions"] = int(match.group(1))
        stats["irrip_pct"] = float(match.group(3))
        stats["switches"] = int(match.group(4))
    match = FRAC.search(text)
    if match:
        stats["frac_mean"] = float(match.group(1))
    for group in GROUP.finditer(text):
        stats[f"miss_pct_{group.group(1)}"] = float(group.group(4))
    return stats


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def write_table(path_stem, header, rows, fmt):
    with open(path_stem + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    with open(path_stem + ".md", "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(i, v) for i, v in enumerate(row)) + " |\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    static = {}
    for key, _ in STATIC:
        for trace in TRACES:
            static[(key, trace)] = parse(os.path.join(HERE, f"{key}_{trace}.txt"))
    lru_ipc = {t: static[("lru", t)]["ipc"] for t in TRACES}

    dyn = {}
    for path in glob.glob(os.path.join(SWEEP_DIR, "*.txt")):
        match = NAME.search(os.path.basename(path))
        if not match:
            continue
        th, n, s, trace = float(match.group(1)), int(match.group(2)), float(match.group(3)), match.group(4)
        if th in THRESHOLDS and n in LEADERS and s in SCALES:
            dyn[(th, n, s, trace)] = parse(path)
    expected = len(THRESHOLDS) * len(LEADERS) * len(SCALES) * len(TRACES)
    if len(dyn) != expected:
        sys.exit(f"found {len(dyn)} sweep results, expected {expected}")
    for st in dyn.values():
        pass
    for key, st in dyn.items():
        st["speedup"] = st["ipc"] / lru_ipc[key[3]]
    for key, st in static.items():
        st["speedup"] = st["ipc"] / lru_ipc[key[1]]

    # ---- per configuration, all traces ----
    rows = []
    for n in LEADERS:
        for s in SCALES:
            for th in THRESHOLDS:
                for trace in TRACES:
                    st = dyn[(th, n, s, trace)]
                    rows.append([th, n, s, trace, st["ipc"], st["speedup"], st["load_mpki"],
                                 st["instr_mpki"], st["data_mpki"], st["irrip_pct"], st["switches"]])
    header = ["threshold", "decision_sets", "scale", "trace", "IPC", "speedup", "L2C_load_MPKI",
              "L2_instr_MPKI", "L2_data_MPKI", "pct_IRRIP_decisions", "mode_switches"]
    fmts = [None, None, None, None, "{:.5f}", "{:.4f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.1f}", "{}"]
    write_table(os.path.join(OUT_DIR, "dyn_sweep_all"), header, rows,
                lambda i, v: fmts[i].format(v) if fmts[i] else (f"{v:g}" if isinstance(v, float) else str(v)))

    # ---- geomean per configuration ----
    summary = {}
    grows = []
    for n in LEADERS:
        for s in SCALES:
            for th in THRESHOLDS:
                sts = [dyn[(th, n, s, t)] for t in TRACES]
                g = {
                    "speedup": geomean([x["speedup"] for x in sts]),
                    "load_mpki": sum(x["load_mpki"] for x in sts) / 3,
                    "instr_mpki": sum(x["instr_mpki"] for x in sts) / 3,
                    "data_mpki": sum(x["data_mpki"] for x in sts) / 3,
                    "irrip_pct": [x["irrip_pct"] for x in sts],
                }
                summary[(th, n, s)] = g
                grows.append([th, n, s, g["speedup"], g["load_mpki"], g["instr_mpki"], g["data_mpki"]]
                             + g["irrip_pct"])
    write_table(os.path.join(OUT_DIR, "dyn_sweep_geomean"),
                ["threshold", "decision_sets", "scale", "geomean_speedup", "mean_L2C_load_MPKI",
                 "mean_L2_instr_MPKI", "mean_L2_data_MPKI", "pct_IRRIP_T1", "pct_IRRIP_T2", "pct_IRRIP_T3"],
                grows,
                lambda i, v: (f"{v:g}" if i < 3 else "{:.4f}".format(v) if i == 3 else
                              "{:.3f}".format(v) if i < 7 else "{:.1f}".format(v)))

    static_g = {key: geomean([static[(key, t)]["speedup"] for t in TRACES]) for key, _ in STATIC}
    best_cfg = max(summary, key=lambda c: summary[c]["speedup"])
    bound = geomean([max(static[("rrip", t)]["speedup"], static[("irrip", t)]["speedup"]) for t in TRACES])

    # ---- final configuration vs static policies ----
    frows = []
    for trace in TRACES:
        for key, label in STATIC:
            st = static[(key, trace)]
            frows.append([trace, label, st["ipc"], st["speedup"], st["load_mpki"], st["instr_mpki"], st["data_mpki"]])
        st = dyn[best_cfg + (trace,)]
        frows.append([trace, "Dyn. I-RRIP", st["ipc"], st["speedup"], st["load_mpki"], st["instr_mpki"], st["data_mpki"]])
    write_table(os.path.join(OUT_DIR, "final_config"),
                ["trace", "policy", "IPC", "speedup", "L2C_load_MPKI", "L2_instr_MPKI", "L2_data_MPKI"],
                frows,
                lambda i, v: str(v) if i < 2 else "{:.5f}".format(v) if i == 2 else
                "{:.4f}".format(v) if i == 3 else "{:.3f}".format(v))

    # ---- plots ----
    colors = {0.5: "#1f77b4", 1.0: "#ff7f0e", 2.0: "#2ca02c"}
    markers = {0.5: "o", 1.0: "s", 2.0: "^"}

    def sweep_plot(metric, ylabel, title, filename, per_trace, refs):
        """One panel per number of decision sets; lines = code scaling; x = threshold."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
        for ax, n in zip(axes, LEADERS):
            for s in SCALES:
                ys = [per_trace(th, n, s) for th in THRESHOLDS]
                ax.plot(THRESHOLDS, ys, marker=markers[s], color=colors[s], label=f"scale s = {s:g}")
            for value, label, style in refs:
                ax.axhline(value, color="gray", linestyle=style, linewidth=1, label=label)
            ax.set_title(f"{n} decision sets per policy")
            ax.set_xlabel("threshold $\\theta$")
            ax.set_xticks(THRESHOLDS)
            ax.grid(linestyle=":", alpha=0.6)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, filename)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    outputs = []
    outputs.append(sweep_plot("speedup", "geomean speedup over LRU", "Dynamic I-RRIP: geomean speedup over LRU (3 traces)",
                              "task2b_speedup_vs_threshold.png",
                              lambda th, n, s: summary[(th, n, s)]["speedup"],
                              [(static_g["irrip"], "I-RRIP (static)", "--"), (static_g["rrip"], "SRRIP", ":"),
                               (bound, "best of SRRIP / I-RRIP per trace", "-.")]))
    outputs.append(sweep_plot("instr", "mean L2 instruction MPKI", "Dynamic I-RRIP: L2 instruction load MPKI (mean of 3 traces)",
                              "task2b_instr_mpki_vs_threshold.png",
                              lambda th, n, s: summary[(th, n, s)]["instr_mpki"],
                              [(sum(static[("irrip", t)]["instr_mpki"] for t in TRACES) / 3, "I-RRIP (static)", "--"),
                               (sum(static[("rrip", t)]["instr_mpki"] for t in TRACES) / 3, "SRRIP", ":")]))
    outputs.append(sweep_plot("data", "mean L2 data MPKI", "Dynamic I-RRIP: L2 data load MPKI (mean of 3 traces)",
                              "task2b_data_mpki_vs_threshold.png",
                              lambda th, n, s: summary[(th, n, s)]["data_mpki"],
                              [(sum(static[("irrip", t)]["data_mpki"] for t in TRACES) / 3, "I-RRIP (static)", "--"),
                               (sum(static[("rrip", t)]["data_mpki"] for t in TRACES) / 3, "SRRIP", ":")]))

    # threshold and scale combine into one cut-off on the unscaled fraction f:
    # s*f / (s*f + 1 - f) > theta  <=>  f > theta / (s*(1 - theta) + theta)   (ignoring the +1)
    def effective(th, s):
        return th / (s * (1 - th) + th)

    fig, ax = plt.subplots(figsize=(8, 5))
    leader_markers = {16: "o", 32: "s", 64: "^"}
    for n in LEADERS:
        for s in SCALES:
            xs = [effective(th, s) for th in THRESHOLDS]
            ys = [summary[(th, n, s)]["speedup"] for th in THRESHOLDS]
            ax.scatter(xs, ys, marker=leader_markers[n], color=colors[s], s=30,
                       label=f"{n} sets, s = {s:g}")
    ax.axhline(static_g["irrip"], color="gray", linestyle="--", linewidth=1, label="I-RRIP (static)")
    ax.axhline(static_g["rrip"], color="gray", linestyle=":", linewidth=1, label="SRRIP")
    ax.axhline(bound, color="gray", linestyle="-.", linewidth=1, label="best of SRRIP / I-RRIP per trace")
    ax.set_xlabel("effective threshold on the unscaled fraction $f$:  $\\theta / (s(1-\\theta)+\\theta)$")
    ax.set_ylabel("geomean speedup over LRU")
    ax.set_title("Dynamic I-RRIP: all 45 configurations against the effective threshold")
    ax.grid(linestyle=":", alpha=0.6)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "task2b_speedup_vs_effective_threshold.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    outputs.append(out)

    # fraction of I-RRIP decisions per trace (scale 1, all leader counts)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    trace_colors = {"Trace1": "#d62728", "Trace2": "#9467bd", "Trace3": "#8c564b"}
    for ax, n in zip(axes, LEADERS):
        for trace in TRACES:
            for s, style in ((0.5, ":"), (1.0, "-"), (2.0, "--")):
                ys = [dyn[(th, n, s, trace)]["irrip_pct"] for th in THRESHOLDS]
                ax.plot(THRESHOLDS, ys, linestyle=style, marker="o", markersize=3, color=trace_colors[trace],
                        label=f"{trace}, s = {s:g}")
        ax.set_title(f"{n} decision sets per policy")
        ax.set_xlabel("threshold $\\theta$")
        ax.set_xticks(THRESHOLDS)
        ax.grid(linestyle=":", alpha=0.6)
    axes[0].set_ylabel("decisions choosing I-RRIP (%)")
    axes[0].legend(fontsize=7, ncol=1)
    fig.suptitle("Dynamic I-RRIP: share of epochs in I-RRIP mode")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "task2b_irrip_fraction.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    outputs.append(out)

    # final configuration: grouped bars like Task 2A
    labels = [label for _, label in STATIC] + ["Dyn. I-RRIP"]
    bar_colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]
    for metric, ylabel, filename in (("ipc", "IPC", "task2b_final_ipc.png"),
                                     ("speedup", "speedup over LRU", "task2b_final_speedup.png"),
                                     ("load_mpki", "L2C load MPKI", "task2b_final_l2c_load_mpki.png"),
                                     ("instr_mpki", "L2 instruction load MPKI", "task2b_final_l2c_instruction_mpki.png"),
                                     ("data_mpki", "L2 data load MPKI", "task2b_final_l2c_data_mpki.png")):
        fig, ax = plt.subplots(figsize=(8, 5))
        width = 0.8 / len(labels)
        allv = []
        for i, label in enumerate(labels):
            vals = []
            for trace in TRACES:
                if label == "Dyn. I-RRIP":
                    vals.append(dyn[best_cfg + (trace,)][metric])
                else:
                    key = [k for k, l in STATIC if l == label][0]
                    vals.append(static[(key, trace)][metric])
            allv += vals
            pos = [x + (i - (len(labels) - 1) / 2) * width for x in range(len(TRACES))]
            bars = ax.bar(pos, vals, width, label=label, color=bar_colors[i])
            fmt = "{:.4f}" if metric in ("ipc", "speedup") else "{:.2f}"
            ax.bar_label(bars, labels=[fmt.format(v) for v in vals], fontsize=7, padding=2, rotation=90)
        if metric == "speedup":
            lo, hi = min(allv), max(allv)
            ax.set_ylim(lo - max((hi - lo) * 0.5, 0.005), hi + max((hi - lo) * 0.6, 0.008))
            ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        else:
            ax.set_ylim(0, max(allv) * 1.2)
        ax.set_xticks(range(len(TRACES)))
        ax.set_xticklabels(TRACES)
        ax.set_ylabel(ylabel)
        th, n, s = best_cfg
        ax.set_title(f"{ylabel}: final dynamic I-RRIP ($\\theta$={th:g}, {n} sets, s={s:g})")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, filename)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        outputs.append(out)

    os.makedirs(REPORT_FIG_DIR, exist_ok=True)
    for out in outputs:
        shutil.copy(out, REPORT_FIG_DIR)

    print(f"static geomean speedup: " + ", ".join(f"{l} {static_g[k]:.4f}" for k, l in STATIC))
    print(f"best-of-two bound: {bound:.4f}")
    print(f"best configuration: threshold {best_cfg[0]:g}, {best_cfg[1]} sets, scale {best_cfg[2]:g}: "
          f"{summary[best_cfg]['speedup']:.4f}")
    print(f"wrote tables and {len(outputs)} plots to {OUT_DIR} (plots copied to report/figures)")


if __name__ == "__main__":
    main()
