#!/usr/bin/env python3
"""
parse_results.py - turn every ChampSim output in results_pf*/ into one table.

Usage:
    python3 artifacts/parse_results.py                 # writes results_summary.csv
    python3 artifacts/parse_results.py --out my.csv    # custom output path
    python3 artifacts/parse_results.py --quiet         # CSV only, no printed table

The CSV is REGENERATED from scratch on every run (never appended), so re-running
is always safe and there are no duplicate rows.

Layout it expects:
    results_pf<T>/<label>.txt     T = trace number, label = config name
    results_pf<T>/baseline.txt    needed for speedup and coverage of trace T
Files ending in .tmp or .FAILED.txt are ignored.

Columns (raw counts first, then derived ones):
    trace, label, status
    ipc, instructions, cycles
    l1d_load_access, l1d_load_miss, l1d_load_mpki
    pf_requested        prefetch_line() calls
    pf_issued           accepted into the L1D prefetch queue
    pf_lookups          reached the L1D tag lookup ("L1D PREFETCH ACCESS")
    pf_already_cached   lookups that found the line already in L1D (wasted)
    pf_fetched          actually fetched from L2 ("ISSUED TO LOWER LEVEL")
    pf_useful, pf_useless, pf_late
    pf_fill_l1, pf_fill_l2          fill level chosen (same / different fill-origin)
    l2c_load_mpki, l2c_data_load_mpki, l2c_instr_load_mpki, llc_load_mpki
    speedup             ipc / baseline ipc (same trace)
    coverage            (baseline load misses - load misses) / baseline load misses
    accuracy            pf_useful / pf_fetched   (same formula as ChampSim)
    late_frac           pf_late / (pf_useful + pf_late)
    wasted_lookup_frac  pf_already_cached / pf_lookups
"""

import argparse
import csv
import glob
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_INSTR = 25_000_000
ROI_MARKER = "Region of Interest Statistics"

NUM = r"(-?nan|-?[\d.]+(?:e[-+]?\d+)?)"
# metric name -> (regex, group index). All regexes run on the ROI part only.
PATTERNS = {
    "ipc":                (rf"^CPU 0 cumulative IPC:\s+{NUM}", 1),
    "instructions":       (r"^CPU 0 cumulative IPC:\s+\S+\s+instructions:\s+(\d+)", 1),
    "cycles":             (r"^CPU 0 cumulative IPC:.*cycles:\s+(\d+)", 1),
    "l1d_load_access":    (r"^L1D LOAD\s+ACCESS:\s+(\d+)", 1),
    "l1d_load_miss":      (r"^L1D LOAD\s+ACCESS:.*?MISS:\s+(\d+)", 1),
    "l1d_load_mpki":      (rf"^L1D LOAD\s+ACCESS:.*MPKI:\s+{NUM}", 1),
    "pf_requested":       (r"^L1D PREFETCH\s+REQUESTED:\s+(\d+)", 1),
    "pf_issued":          (r"^L1D PREFETCH\s+REQUESTED:.*?ISSUED:\s+(\d+)", 1),
    "pf_useful":          (r"^L1D PREFETCH\s+REQUESTED:.*?USEFUL:\s+(\d+)", 1),
    "pf_useless":         (r"^L1D PREFETCH\s+REQUESTED:.*?USELESS:\s+(\d+)", 1),
    "pf_lookups":         (r"^L1D PREFETCH\s+ACCESS:\s+(\d+)", 1),
    "pf_already_cached":  (r"^L1D PREFETCH\s+ACCESS:.*?HIT:\s+(\d+)", 1),
    "pf_fetched":         (r"^L1D USEFUL LOAD PREFETCHES:.*?ISSUED TO LOWER LEVEL:\s+(\d+)", 1),
    "pf_late":            (r"^L1D TIMELY PREFETCHES:.*?LATE PREFETCHES:\s+(\d+)", 1),
    "pf_fill_l1":         (r"^L1D PREFETCHES SAME FILL-ORIGIN LEVEL:\s+(\d+)", 1),
    "pf_fill_l2":         (r"^L1D PREFETCHES SAME FILL-ORIGIN LEVEL:.*?DIFFERENT FILL-ORIGIN LEVEL:\s+(\d+)", 1),
    "l2c_load_mpki":      (rf"^L2C LOAD\s+ACCESS:.*MPKI:\s+{NUM}", 1),
    "l2c_data_load_mpki": (rf"^L2C DATA LOAD MPKI:\s+{NUM}", 1),
    "l2c_instr_load_mpki":(rf"^L2C INSTRUCTION LOAD MPKI:\s+{NUM}", 1),
    "llc_load_mpki":      (rf"^LLC LOAD\s+ACCESS:.*MPKI:\s+{NUM}", 1),
}
# Lines that only exist when a prefetcher issued something: missing -> 0.
OPTIONAL_ZERO = {"pf_lookups", "pf_already_cached"}

RAW_COLS = list(PATTERNS)
DERIVED_COLS = ["speedup", "coverage", "accuracy", "late_frac", "wasted_lookup_frac"]
COLUMNS = ["trace", "label", "status"] + RAW_COLS + DERIVED_COLS


def to_num(s):
    if s is None or "nan" in s:
        return None
    return float(s) if any(c in s for c in ".e") else int(s)


def div(a, b):
    return None if a is None or b in (None, 0) else a / b


def parse_file(path):
    """Return a dict of raw metrics plus a 'status' field."""
    with open(path, errors="replace") as f:
        text = f.read()
    row = {k: None for k in RAW_COLS}
    if ROI_MARKER not in text:
        row["status"] = "incomplete (no ROI section)"
        return row
    roi = text[text.index(ROI_MARKER):]
    for key, (pat, grp) in PATTERNS.items():
        m = re.search(pat, roi, re.M)
        row[key] = to_num(m.group(grp)) if m else (0 if key in OPTIONAL_ZERO else None)

    problems = []
    if row["ipc"] is None:
        problems.append("no IPC")
    instr = row["instructions"]
    if instr is not None and not (SIM_INSTR <= instr <= SIM_INSTR + 10):
        problems.append(f"instructions={instr}, expected {SIM_INSTR}")
    if row["l1d_load_miss"] is None:
        problems.append("no L1D LOAD line")
    row["status"] = "ok" if not problems else "; ".join(problems)
    return row


def add_derived(row, base):
    ok = row["status"] == "ok" and base is not None and base["status"] == "ok"
    row["speedup"] = div(row["ipc"], base["ipc"]) if ok else None
    row["coverage"] = (div(base["l1d_load_miss"] - row["l1d_load_miss"], base["l1d_load_miss"])
                       if ok else None)
    row["accuracy"] = div(row["pf_useful"], row["pf_fetched"])
    late, useful = row["pf_late"], row["pf_useful"]
    row["late_frac"] = div(late, (useful or 0) + (late or 0)) if late is not None else None
    row["wasted_lookup_frac"] = div(row["pf_already_cached"], row["pf_lookups"])


def label_order(label):
    # baseline first, ip_stride second, then natural order (spp2 < spp10)
    fixed = {"baseline": (0,), "ip_stride": (1,)}
    if label in fixed:
        return fixed[label]
    return (2,) + tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", label))


def collect():
    rows = []
    for d in sorted(glob.glob(os.path.join(REPO, "results_pf*"))):
        m = re.search(r"results_pf(\d+)$", d)
        if not m or not os.path.isdir(d):
            continue
        trace = int(m.group(1))
        files = [f for f in glob.glob(os.path.join(d, "*.txt")) if not f.endswith(".FAILED.txt")]
        parsed = {os.path.basename(f)[:-4]: parse_file(f) for f in files}
        base = parsed.get("baseline")
        for label in sorted(parsed, key=label_order):
            row = parsed[label]
            add_derived(row, base)
            if base is None and row["status"] == "ok":
                row["status"] = "ok (no baseline.txt: speedup/coverage missing)"
            rows.append({"trace": trace, "label": label, **row})
    return rows


def fmt(v, kind):
    if v is None:
        return "-"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "int":
        return f"{int(v):,}"
    return f"{v:.{ {'ipc': 5, 'x': 4, 'f2': 2}[kind] }f}"


def print_tables(rows):
    cols = [("label", "label", None), ("IPC", "ipc", "ipc"), ("speedup", "speedup", "x"),
            ("L1D ld MPKI", "l1d_load_mpki", "f2"), ("coverage", "coverage", "pct"),
            ("accuracy", "accuracy", "pct"), ("late", "late_frac", "pct"),
            ("fetched", "pf_fetched", "int"), ("useful", "pf_useful", "int"),
            ("requested", "pf_requested", "int"), ("wasted lkp", "wasted_lookup_frac", "pct")]
    for trace in sorted({r["trace"] for r in rows}):
        sub = [r for r in rows if r["trace"] == trace]
        table = [[h for h, _, _ in cols]]
        for r in sub:
            table.append([r["label"] if kind is None else fmt(r[k], kind) for _, k, kind in cols])
        widths = [max(len(row[i]) for row in table) for i in range(len(cols))]
        print(f"\n=== Trace {trace} ===")
        for i, row in enumerate(table):
            print("  ".join(c.ljust(w) if j == 0 else c.rjust(w)
                            for j, (c, w) in enumerate(zip(row, widths))))
            if i == 0:
                print("  ".join("-" * w for w in widths))
        for r in sub:
            if not r["status"].startswith("ok"):
                print(f"  ! {r['label']}: {r['status']}")
            elif "no baseline" in r["status"]:
                print(f"  ! {r['label']}: no baseline.txt in results_pf{trace}")

    # Geometric-mean speedup per label over the traces it has results for.
    all_traces = sorted({r["trace"] for r in rows})
    by_label = {}
    for r in rows:
        if r["speedup"] is not None:
            by_label.setdefault(r["label"], {})[r["trace"]] = r["speedup"]
    if by_label:
        print("\n=== Geomean speedup ===")
        for label in sorted(by_label, key=label_order):
            sp = by_label[label]
            gm = math.exp(sum(math.log(v) for v in sp.values()) / len(sp))
            note = "" if len(sp) == len(all_traces) else \
                f"   (only traces {','.join(map(str, sorted(sp)))} - not comparable to full set)"
            print(f"  {label:<14} {gm:.4f}{note}")


def main():
    ap = argparse.ArgumentParser(description="Summarise results_pf*/ into one CSV.")
    ap.add_argument("--out", default=os.path.join(REPO, "results_summary.csv"),
                    help="output CSV path (default: results_summary.csv in repo root)")
    ap.add_argument("--quiet", action="store_true", help="don't print tables")
    args = ap.parse_args()

    rows = collect()
    if not rows:
        sys.exit("no result files found in results_pf*/")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in COLUMNS})
    if not args.quiet:
        print_tables(rows)
    print(f"\nwrote {len(rows)} rows -> {os.path.relpath(args.out, os.getcwd())}")


if __name__ == "__main__":
    main()
