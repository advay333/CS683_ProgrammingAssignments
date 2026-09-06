#!/usr/bin/env python3
"""Derive metrics and per-point medians from perf_metrics.csv.

Reads the wide CSV written by run_perf_profile.sh and writes

    perf_derived.csv   one row per execution, raw counters + derived metrics
    perf_summary.csv   one row per (label, H, W, K), median over the repetitions

No calibration offset is applied: the calibration run appears as its own label
with the same schema, so any subtraction can be done afterwards from these
files (or from perf_raw_long.csv) without re-running anything.

Medians rather than means: perf runs have a long right tail from stray
interrupts and page-cache activity, and a single bad run moves a 20-sample mean
noticeably while leaving the median alone.  The MAD column is reported so a
point with genuinely unstable counters is still visible.

Stdlib only.
"""

import argparse
import csv
import os
import statistics
import sys

COUNTERS = [
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "branch-instructions",
    "branch-misses",
    "instructions",
    "cpu-cycles",
    "context-switches",
    "page-faults",
]

DERIVED = [
    "L1D_MPKI",          # L1D load misses per kilo-instruction
    "L1D_miss_rate",     # L1D load misses / L1D loads
    "branch_MPKI",       # branch mispredictions per kilo-instruction
    "branch_rate",       # branch instructions / instructions
    "branch_miss_rate",  # branch misses / branch instructions
    "IPC",
    "CPI",
]

KEY = ["label", "stage", "variant", "H", "W", "K"]


def to_float(s):
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def div(a, b, scale=1.0):
    if a is None or b in (None, 0.0):
        return None
    return a / b * scale


def derive(row):
    c = {k: to_float(row.get(k)) for k in COUNTERS}
    return {
        "L1D_MPKI":         div(c["L1-dcache-load-misses"], c["instructions"], 1000.0),
        "L1D_miss_rate":    div(c["L1-dcache-load-misses"], c["L1-dcache-loads"]),
        "branch_MPKI":      div(c["branch-misses"], c["instructions"], 1000.0),
        "branch_rate":      div(c["branch-instructions"], c["instructions"]),
        "branch_miss_rate": div(c["branch-misses"], c["branch-instructions"]),
        "IPC":              div(c["instructions"], c["cpu-cycles"]),
        "CPI":              div(c["cpu-cycles"], c["instructions"]),
    }


def fmt(x, places=4):
    return "" if x is None else f"{x:.{places}g}"


def mad(values, med):
    """Median absolute deviation -- a spread estimate that, unlike the standard
    deviation, is not dragged around by the one run that got descheduled."""
    return statistics.median([abs(v - med) for v in values]) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv", help="perf_metrics.csv from run_perf_profile.sh")
    ap.add_argument("--outdir", default=None, help="where to write (default: alongside the input)")
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.metrics_csv))
    os.makedirs(outdir, exist_ok=True)

    with open(args.metrics_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{args.metrics_csv}: no data rows")

    # ---------------------------------------------------------- per-execution
    derived_path = os.path.join(outdir, "perf_derived.csv")
    base_cols = ["label", "stage", "variant", "H", "W", "K", "seed", "rep",
                 "exit_status", "min_pct_enabled"]
    with open(derived_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(base_cols + COUNTERS + DERIVED)
        for r in rows:
            d = derive(r)
            w.writerow([r.get(c, "") for c in base_cols]
                       + [r.get(c, "") for c in COUNTERS]
                       + [fmt(d[c], 6) for c in DERIVED])

    # ------------------------------------------------------------- aggregate
    groups = {}
    bad = 0
    for r in rows:
        if to_float(r.get("exit_status")) not in (0.0, None):
            bad += 1
            continue
        key = tuple(r[k] for k in KEY)
        groups.setdefault(key, []).append(r)

    summary_path = os.path.join(outdir, "perf_summary.csv")
    cols = COUNTERS + DERIVED
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(KEY + ["n"]
                   + [f"{c}_median" for c in cols]
                   + [f"{c}_mad" for c in cols])
        for key in sorted(groups, key=lambda k: (k[0], int(k[3]))):
            grp = groups[key]
            meds, mads = [], []
            for c in cols:
                vals = [v for v in
                        ((to_float(r.get(c)) if c in COUNTERS else derive(r)[c]) for r in grp)
                        if v is not None]
                if vals:
                    m = statistics.median(vals)
                    meds.append(fmt(m, 8))
                    mads.append(fmt(mad(vals, m), 4))
                else:
                    meds.append("")
                    mads.append("")
            w.writerow(list(key) + [len(grp)] + meds + mads)

    # ------------------------------------------------------------- terminal
    print(f"wrote {derived_path}")
    print(f"wrote {summary_path}")
    if bad:
        print(f"note: {bad} execution(s) with a non-zero exit status were excluded from the medians")

    hdr = f"{'label':<16}{'size':>8}{'MPKI':>10}{'L1D miss':>10}{'br MPKI':>10}{'IPC':>8}{'Ginstr':>10}{'Gcycles':>10}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(groups, key=lambda k: (k[0], int(k[3]))):
        grp = groups[key]

        def med_of(name):
            vals = [v for v in
                    ((to_float(r.get(name)) if name in COUNTERS else derive(r)[name]) for r in grp)
                    if v is not None]
            return statistics.median(vals) if vals else None

        label, _, _, H, W, _ = key
        ins, cyc = med_of("instructions"), med_of("cpu-cycles")
        print(f"{label:<16}{H + 'x' + W:>8}"
              f"{fmt(med_of('L1D_MPKI'), 4):>10}"
              f"{fmt(med_of('L1D_miss_rate'), 3):>10}"
              f"{fmt(med_of('branch_MPKI'), 3):>10}"
              f"{fmt(med_of('IPC'), 3):>8}"
              f"{'' if ins is None else f'{ins / 1e9:.3f}':>10}"
              f"{'' if cyc is None else f'{cyc / 1e9:.3f}':>10}")


if __name__ == "__main__":
    main()