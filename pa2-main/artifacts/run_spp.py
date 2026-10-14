#!/usr/bin/env python3
"""
run_spp.py - build the current SPP prefetcher and simulate it on chosen traces.

Usage (run from anywhere; paths are resolved relative to the repo root):
    python3 artifacts/run_spp.py NAME TRACE [TRACE ...] [--note "..."] [--jobs N] [--force]

Examples:
    python3 artifacts/run_spp.py spp4 1 2 3 --note "uniform fast path added"
    python3 artifacts/run_spp.py spp4 1 --jobs 1

What it does, in order:
    1. Safety checks: refuses if NAME was already used (snapshot or result file
       exists) unless --force is given.
    2. Snapshots prefetcher/spp.l1d_pref -> SPP_prefetchers/NAME.l1d_pref
       (taken BEFORE building, so the snapshot is exactly what gets compiled).
    3. Builds:  bash build_champsim.sh spp lru NAME   (builds are never parallel)
    4. Appends a line to runsTrackers.txt:  NAME - note  [traces ..., date]
    5. Runs the simulation(s) with 25M warmup + 25M simulation instructions.
       Output goes to results_pf<T>/NAME.txt. Each run writes to NAME.txt.tmp
       first and is renamed only on success; a failed/interrupted run is kept
       as NAME.FAILED.txt so the parser ignores it.
"""

import argparse
import datetime
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- settings ---
WARMUP_INSTR = 25_000_000
SIM_INSTR = 25_000_000
VALID_TRACES = (1, 2, 3)
RESERVED_NAMES = {"baseline", "ip_stride"}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPP_SRC = os.path.join(REPO, "prefetcher", "spp.l1d_pref")
SNAP_DIR = os.path.join(REPO, "SPP_prefetchers")
TRACKER = os.path.join(REPO, "runsTrackers.txt")
BIN_FMT = "hashed_perceptron-no-spp-no-no-no-no-no-lru-lru-lru-lru-lru-lru-lru-lru-1core-{}"

_running = []                 # live simulation processes (for Ctrl-C cleanup)
_running_lock = threading.Lock()
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def trace_path(t):
    return os.path.join(REPO, "Trace_Files", f"Trace{t}.gz")


def result_path(t, name, suffix=".txt"):
    return os.path.join(REPO, f"results_pf{t}", name + suffix)


# ------------------------------------------------------------------ checks ---
def preflight(name, traces, force):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        die("NAME may only contain letters, digits, '_', '-' and '.'")
    if name in RESERVED_NAMES:
        die(f"'{name}' is reserved for the reference results")
    if not os.path.isfile(SPP_SRC):
        die(f"{SPP_SRC} not found")
    for t in traces:
        if not os.path.isfile(trace_path(t)):
            die(f"trace not found: {trace_path(t)}")

    clashes = []
    snap = os.path.join(SNAP_DIR, name + ".l1d_pref")
    if os.path.exists(snap):
        clashes.append(os.path.relpath(snap, REPO))
    for t in traces:
        if os.path.exists(result_path(t, name)):
            clashes.append(os.path.relpath(result_path(t, name), REPO))
    if clashes and not force:
        die("name already used, refusing to overwrite:\n  "
            + "\n  ".join(clashes)
            + "\nPick a new NAME, or pass --force to overwrite.")
    if clashes:
        log("--force given: overwriting " + ", ".join(clashes))


# --------------------------------------------------------- snapshot/build ---
def snapshot(name):
    os.makedirs(SNAP_DIR, exist_ok=True)
    dst = os.path.join(SNAP_DIR, name + ".l1d_pref")
    shutil.copy2(SPP_SRC, dst)
    log(f"snapshot: prefetcher/spp.l1d_pref -> {os.path.relpath(dst, REPO)}")
    return dst


def build(name, snap):
    binary = os.path.join(REPO, "bin", BIN_FMT.format(name))
    if os.path.exists(binary):
        os.remove(binary)          # so a failed build can't leave a stale binary
    log(f"building: bash build_champsim.sh spp lru {name}  (takes ~1 min)")
    t0 = time.time()
    proc = subprocess.run(["bash", "build_champsim.sh", "spp", "lru", name],
                          cwd=REPO, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not os.path.isfile(binary):
        logfile = os.path.join(SNAP_DIR, name + ".build_error.log")
        with open(logfile, "w") as f:
            f.write(proc.stdout)
        os.remove(snap)            # name stays free for the next attempt
        tail = "\n".join(proc.stdout.strip().splitlines()[-25:])
        die(f"build failed (exit {proc.returncode}). Last lines:\n{tail}\n"
            f"Full log: {os.path.relpath(logfile, REPO)}")
    log(f"build OK in {time.time() - t0:.0f}s -> bin/{os.path.basename(binary)}")
    return binary


def record_run(name, traces, note):
    line = (f"{name} - {note or '(no note)'}  "
            f"[traces {','.join(map(str, traces))}, "
            f"{datetime.datetime.now():%Y-%m-%d %H:%M}]")
    needs_nl = False
    if os.path.exists(TRACKER) and os.path.getsize(TRACKER) > 0:
        with open(TRACKER, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_nl = f.read(1) != b"\n"
    with open(TRACKER, "a") as f:
        f.write(("\n" if needs_nl else "") + line + "\n")
    log(f"recorded in runsTrackers.txt: {line}")


# -------------------------------------------------------------- simulate ---
def simulate(binary, name, t):
    final = result_path(t, name)
    tmp = result_path(t, name, ".txt.tmp")
    failed = result_path(t, name, ".FAILED.txt")
    os.makedirs(os.path.dirname(final), exist_ok=True)
    for stale in (tmp, failed):
        if os.path.exists(stale):
            os.remove(stale)

    cmd = [binary,
           "-warmup_instructions", str(WARMUP_INSTR),
           "-simulation_instructions", str(SIM_INSTR),
           "-traces", f"./Trace_Files/Trace{t}.gz"]
    log(f"trace {t}: started")
    t0 = time.time()
    with open(tmp, "w") as out:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT)
        with _running_lock:
            _running.append((proc, tmp, failed))
        rc = proc.wait()
        with _running_lock:
            _running.remove((proc, tmp, failed))

    with open(tmp, errors="replace") as f:
        finished = "Region of Interest Statistics" in f.read()
    mins = (time.time() - t0) / 60
    if rc == 0 and finished:
        os.replace(tmp, final)
        log(f"trace {t}: done in {mins:.1f} min -> {os.path.relpath(final, REPO)}")
        return True
    os.replace(tmp, failed)
    log(f"trace {t}: FAILED (exit code {rc}) after {mins:.1f} min "
        f"-> {os.path.relpath(failed, REPO)}")
    return False


def kill_running():
    with _running_lock:
        items = list(_running)
    for proc, tmp, failed in items:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if os.path.exists(tmp):
            os.replace(tmp, failed)


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(
        description="Build prefetcher/spp.l1d_pref as NAME and simulate it.")
    ap.add_argument("name", help="name for this SPP version, e.g. spp4")
    ap.add_argument("traces", nargs="+", type=int, choices=VALID_TRACES,
                    help="trace number(s) to run: 1 2 3")
    ap.add_argument("--note", default="", help="description saved to runsTrackers.txt")
    ap.add_argument("--jobs", type=int, default=1,
                    help="simulations to run at the same time (default 1 = sequential)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing snapshot/results with the same NAME")
    args = ap.parse_args()
    traces = sorted(set(args.traces))

    preflight(args.name, traces, args.force)
    snap = snapshot(args.name)
    binary = build(args.name, snap)
    record_run(args.name, traces, args.note)

    jobs = max(1, min(args.jobs, len(traces)))
    log(f"simulating {len(traces)} trace(s), {jobs} at a time, "
        f"{WARMUP_INSTR // 10**6}M warmup + {SIM_INSTR // 10**6}M sim")
    pool = ThreadPoolExecutor(max_workers=jobs)
    try:
        ok = list(pool.map(lambda t: simulate(binary, args.name, t), traces))
    except KeyboardInterrupt:
        log("interrupted: stopping simulations")
        pool.shutdown(wait=False, cancel_futures=True)
        kill_running()
        sys.exit(130)
    pool.shutdown()

    if all(ok):
        log("all runs finished. Next: python3 artifacts/parse_results.py")
    else:
        bad = [t for t, good in zip(traces, ok) if not good]
        die(f"run(s) failed for trace(s) {bad}; see the .FAILED.txt files")


if __name__ == "__main__":
    main()
