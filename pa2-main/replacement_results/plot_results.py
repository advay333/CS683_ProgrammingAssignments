#!/usr/bin/env python3
"""Parse ChampSim outputs in this directory and plot IPC, speedup and L2C MPKI.

Expects files named <policy>_<trace>.txt (e.g. lru_Trace1.txt, irrip_Trace2.txt)
next to this script. Writes a summary table (CSV + Markdown) and one PNG per
metric into the same directory.
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

POLICIES = ["lru", "rrip", "irrip"]
POLICY_LABELS = {"lru": "LRU", "rrip": "SRRIP", "irrip": "I-RRIP"}
TRACES = ["Trace1", "Trace2", "Trace3"]
BASELINE = "lru"  # speedup is IPC / IPC of this policy

# Regexes for the final (ROI) stats ChampSim prints at the end of a run.
PATTERNS = {
    "ipc": re.compile(r"^CPU 0 cumulative IPC:\s*([\d.]+)", re.M),
    "l2c_load_mpki": re.compile(r"^L2C LOAD\s+ACCESS:.*MPKI:\s*([\d.]+)", re.M),
    "l2c_instr_mpki": re.compile(r"^L2C INSTRUCTION LOAD MPKI:\s*([\d.]+)", re.M),
    "l2c_data_mpki": re.compile(r"^L2C DATA LOAD MPKI:\s*([\d.]+)", re.M),
}

METRICS = [
    # key, title, y-axis label, output file
    ("ipc", "IPC", "IPC (higher is better)", "ipc.png"),
    ("speedup", "Speedup over LRU", "Speedup (higher is better)", "speedup.png"),
    ("l2c_load_mpki", "L2C Load MPKI", "MPKI (lower is better)", "l2c_load_mpki.png"),
    ("l2c_instr_mpki", "L2C Instruction Load MPKI", "MPKI (lower is better)", "l2c_instruction_mpki.png"),
    ("l2c_data_mpki", "L2C Data Load MPKI", "MPKI (lower is better)", "l2c_data_mpki.png"),
]


def parse_file(path):
    with open(path, errors="replace") as f:
        text = f.read()
    stats = {}
    for key, pattern in PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"'{key}' not found in {os.path.basename(path)} (did the run finish?)")
        stats[key] = float(match.group(1))
    return stats


def collect():
    data = {}
    for policy in POLICIES:
        for trace in TRACES:
            path = os.path.join(RESULTS_DIR, f"{policy}_{trace}.txt")
            if not os.path.exists(path):
                sys.exit(f"Missing result file: {path}")
            data[(policy, trace)] = parse_file(path)
    for policy in POLICIES:
        for trace in TRACES:
            data[(policy, trace)]["speedup"] = (
                data[(policy, trace)]["ipc"] / data[(BASELINE, trace)]["ipc"]
            )
    return data


def write_tables(data):
    columns = [(key, title) for key, title, _, _ in METRICS]

    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("trace,policy," + ",".join(key for key, _ in columns) + "\n")
        for trace in TRACES:
            for policy in POLICIES:
                values = ",".join(f"{data[(policy, trace)][key]:.6f}" for key, _ in columns)
                f.write(f"{trace},{POLICY_LABELS[policy]},{values}\n")

    header = "| Trace | Policy | " + " | ".join(title for _, title in columns) + " |"
    divider = "|" + "---|" * (len(columns) + 2)
    lines = [header, divider]
    for trace in TRACES:
        for policy in POLICIES:
            s = data[(policy, trace)]
            lines.append(
                f"| {trace} | {POLICY_LABELS[policy]} | {s['ipc']:.5f} | {s['speedup']:.4f} | "
                f"{s['l2c_load_mpki']:.3f} | {s['l2c_instr_mpki']:.3f} | {s['l2c_data_mpki']:.3f} |"
            )
    table = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write(table + "\n")

    print(table)
    print(f"\nTables written to summary.csv and summary.md in {RESULTS_DIR}")


def plot_metric(data, key, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(POLICIES)
    width = 0.8 / n
    x = range(len(TRACES))

    values_all = []
    for i, policy in enumerate(POLICIES):
        values = [data[(policy, trace)][key] for trace in TRACES]
        values_all.extend(values)
        positions = [xi + (i - (n - 1) / 2) * width for xi in x]
        bars = ax.bar(positions, values, width, label=POLICY_LABELS[policy])
        fmt = "{:.4f}" if key in ("ipc", "speedup") else "{:.2f}"
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], fontsize=8, padding=2)

    if key == "speedup":
        # Differences are small; zoom in around 1.0 so they are visible.
        lo, hi = min(values_all), max(values_all)
        margin = max((hi - lo) * 0.5, 0.005)
        ax.set_ylim(lo - margin, hi + margin)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    else:
        ax.set_ylim(0, max(values_all) * 1.15)

    ax.set_xticks(list(x))
    ax.set_xticklabels(TRACES)
    ax.set_xlabel("Trace")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    data = collect()
    write_tables(data)
    for key, title, ylabel, filename in METRICS:
        plot_metric(data, key, title, ylabel, filename)


if __name__ == "__main__":
    main()
