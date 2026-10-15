#!/usr/bin/env python3
"""Parameter sweep for the dynamic I-RRIP L2C replacement policies.

For every (threshold, decision sets, code scaling) combination this script
  1. writes a copy of replacement/<policy>.l2c_repl with the three global
     variables changed into replacement/l2c_replacement.cc,
  2. rebuilds ChampSim (incrementally, only the L2C replacement file changes),
  3. keeps the binary as bin/<policy>_t<th>_l<sets>_s<scale>,
  4. runs it on every trace, writing
     replacement_results/<policy>/<policy>_t<th>_l<sets>_s<scale>_<trace>.txt

Builds are sequential (they share obj/), simulations run in parallel.
Finished result files are skipped, so an interrupted sweep can be restarted.

Safety checks:
  - a lock file stops a second sweep (or build_champsim.sh run through this
    script) from sharing obj/ and l2c_replacement.cc at the same time,
  - policy files are read once at start, so editing them mid-sweep has no effect,
  - every build must recompile l2c_replacement.cc from exactly the generated source,
  - every result must be a complete run with the requested instruction counts,
    the expected L2C policy banner and, for stats builds, the expected parameters,
  - results are only written under their final name once all checks pass.

Run it from WSL/Linux inside pa2-main.
"""

import argparse
import fcntl
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))

THRESHOLDS = [0.15, 0.25, 0.45, 0.65, 0.75]
DECISION_SETS = [16, 32, 64]
SCALES = [0.5, 1.0, 2.0]
TRACES = ["Trace1", "Trace2", "Trace3"]

L2C_SET = 1024
DEFAULT_JOBS = 8
MIN_FREE_MB_PER_JOB = 200  # a run uses ~35 MB plus its gunzip pipe; keep a wide margin

# same defaults build_champsim.sh uses for everything except the L2C policy
DEFAULT_COPIES = [
    ("branch/hashed_perceptron.bpred", "branch/branch_predictor.cc"),
    ("prefetcher/no.l1i_pref", "prefetcher/l1i_prefetcher.cc"),
    ("prefetcher/no.l1d_pref", "prefetcher/l1d_prefetcher.cc"),
    ("prefetcher/no.l2c_pref", "prefetcher/l2c_prefetcher.cc"),
    ("prefetcher/no.llc_pref", "prefetcher/llc_prefetcher.cc"),
    ("prefetcher/no.itlb_pref", "prefetcher/itlb_prefetcher.cc"),
    ("prefetcher/no.dtlb_pref", "prefetcher/dtlb_prefetcher.cc"),
    ("prefetcher/no.stlb_pref", "prefetcher/stlb_prefetcher.cc"),
    ("replacement/lru.btb_repl", "replacement/btb_replacement.cc"),
    ("replacement/lru.l1i_repl", "replacement/l1i_replacement.cc"),
    ("replacement/lru.l1d_repl", "replacement/l1d_replacement.cc"),
    ("replacement/lru.llc_repl", "replacement/llc_replacement.cc"),
    ("replacement/lru.itlb_repl", "replacement/itlb_replacement.cc"),
    ("replacement/lru.dtlb_repl", "replacement/dtlb_replacement.cc"),
    ("replacement/lru.stlb_repl", "replacement/stlb_replacement.cc"),
]

L2C_FILE = "replacement/l2c_replacement.cc"
L2C_OBJ = "obj/replacement/l2c_replacement.o"
LOCK_FILE = "replacement_results/.sweep.lock"
MANIFEST = "replacement_results/sweep_manifest.csv"

# the three globals in dyn_rrip*.l2c_repl that the sweep changes
GLOBAL_PATTERNS = {
    "threshold": re.compile(r"^(\s*float\s+threshold\s*=\s*)[^;]+;", re.M),
    "sets": re.compile(r"^(\s*uint32_t\s+num_decision_sets\s*=\s*)[^;]+;", re.M),
    "scale": re.compile(r"^(\s*float\s+code_scaling\s*=\s*)[^;]+;", re.M),
}
BANNER = re.compile(r'cout\s*<<\s*"(Initialize[^"]*L2C)"')
DYN_PARAMS = re.compile(r"^DYN_PARAMS threshold: (\S+) decision_sets: (\d+) code_scaling: (\S+)", re.M)


def fmt(value):
    """0.15 -> '0.15', 1.0 -> '1', used in file and binary names."""
    return f"{value:g}"


def config_name(policy, threshold, sets, scale):
    return f"{policy}_t{fmt(threshold)}_l{sets}_s{fmt(scale)}"


def make_source(template, threshold, sets, scale):
    values = {
        "threshold": f"{float(threshold)!r}",
        "sets": f"{int(sets)}",
        "scale": f"{float(scale)!r}",
    }
    for key, pattern in GLOBAL_PATTERNS.items():
        template, count = pattern.subn(lambda m: m.group(1) + values[key] + ";", template)
        if count != 1:
            sys.exit(f"Expected exactly one '{key}' global in the policy file, found {count}")
    return template


def run(cmd, log):
    """Run cmd, append its output to log, and return (exit code, output)."""
    result = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n{result.stdout}")
    return result.returncode, result.stdout


def acquire_lock():
    path = os.path.join(ROOT, LOCK_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("Another sweep is already running in this directory (lock held on "
                 f"{LOCK_FILE}). Wait for it to finish.")
    handle.write(str(os.getpid()))
    handle.flush()
    return handle  # keep open: the lock lasts as long as this process


def free_memory_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def prepare_tree(log):
    for src, dst in DEFAULT_COPIES:
        shutil.copyfile(os.path.join(ROOT, src), os.path.join(ROOT, dst))
    os.makedirs(os.path.join(ROOT, "bin"), exist_ok=True)
    code, _ = run(["make", "clean"], log)
    if code != 0:
        sys.exit(f"'make clean' failed, see {log}")


def build(source, name, log):
    src_path = os.path.join(ROOT, L2C_FILE)
    with open(src_path, "w", newline="") as f:
        f.write(source)
    with open(src_path, newline="") as f:
        if f.read() != source:
            sys.exit(f"{L2C_FILE} does not contain the generated source for {name}")

    built = os.path.join(ROOT, "bin", "champsim")
    if os.path.exists(built):
        os.remove(built)
    # delete the old object so make always recompiles it; timestamps on /mnt/c
    # can be skewed relative to WSL's clock, so mtimes are not trusted
    obj = os.path.join(ROOT, L2C_OBJ)
    if os.path.exists(obj):
        os.remove(obj)
    code, output = run(["make"], log)
    if code != 0 or not os.path.exists(built):
        sys.exit(f"Build of {name} failed, see {log}")

    # make must have recompiled the L2C policy from the file written above
    if f"Compiling {L2C_FILE}" not in output or not os.path.exists(obj):
        sys.exit(f"make did not recompile {L2C_FILE} for {name}, see {log}")
    with open(src_path, newline="") as f:
        if f.read() != source:
            sys.exit(f"{L2C_FILE} changed while {name} was being built")

    target = os.path.join(ROOT, "bin", name)
    shutil.move(built, target)
    return target


def finished(path, warmup, sim):
    """A result counts as done only if it is complete and used the requested instruction counts."""
    if not os.path.exists(path):
        return False
    with open(path, errors="replace") as f:
        text = f.read()
    return ("Region of Interest Statistics" in text and "L2C INSTRUCTION LOAD MPKI" in text
            and f"Warmup Instructions: {warmup}" in text and f"Simulation Instructions: {sim}" in text)


def check_output(text, banner, warmup, sim, threshold, sets, scale):
    """Return None if the run output is complete and matches the config, else the problem."""
    if "Region of Interest Statistics" not in text or "L2C INSTRUCTION LOAD MPKI" not in text:
        return "output is incomplete"
    if f"Warmup Instructions: {warmup}" not in text or f"Simulation Instructions: {sim}" not in text:
        return "warmup/simulation instruction counts do not match the request"
    if not re.search(r"^CPU 0 cumulative IPC: [\d.]+ instructions: \d+ cycles: \d+", text, re.M):
        return "no final IPC line"
    if banner and banner not in text:
        return f"L2C policy banner '{banner}' missing (wrong binary?)"
    params = DYN_PARAMS.search(text)
    if params:
        got = (float(params.group(1)), int(params.group(2)), float(params.group(3)))
        if (abs(got[0] - threshold) > 1e-4 or got[1] != sets or abs(got[2] - scale) > 1e-4):
            return f"binary reports parameters {got}, expected {(threshold, sets, scale)}"
    return None


RUNNING = set()        # simulations in flight, so a stop signal can terminate them
RUNNING_LOCK = threading.Lock()
STOPPING = threading.Event()


def simulate(binary, trace, out_path, warmup, sim, banner, threshold, sets, scale):
    if STOPPING.is_set():
        return out_path, "sweep stopped before this run started"
    partial = out_path + ".partial"
    cmd = [binary, "-warmup_instructions", str(warmup), "-simulation_instructions", str(sim),
           "-traces", os.path.join("Trace_Files", f"{trace}.gz")]
    with open(partial, "w") as f:
        # own process group, so terminating it also stops the gunzip it spawns
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        with RUNNING_LOCK:
            RUNNING.add(proc)
        try:
            code = proc.wait()
        finally:
            with RUNNING_LOCK:
                RUNNING.discard(proc)
    if STOPPING.is_set():
        return out_path, f"sweep stopped; output kept in {partial}"
    with open(partial, errors="replace") as f:
        text = f.read()
    problem = f"exit code {code}" if code != 0 else check_output(text, banner, warmup, sim, threshold, sets, scale)
    if problem is None:
        os.replace(partial, out_path)
        return out_path, None
    return out_path, f"{problem}; output kept in {partial}"


def stop_sweep(signum, frame):
    """SIGTERM/SIGINT: stop every running simulation, then let main()'s cleanup run."""
    if STOPPING.is_set():
        return
    STOPPING.set()
    print(f"\nReceived signal {signum}, stopping all simulations...", flush=True)
    with RUNNING_LOCK:
        procs = list(RUNNING)
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
    raise SystemExit(130)


def main():
    # log lines appear immediately even when stdout is redirected to a file
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, stop_sweep)
    signal.signal(signal.SIGINT, stop_sweep)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", nargs="+", default=["dyn_rrip"],
                        help="policy names from replacement/<name>.l2c_repl (default: dyn_rrip)")
    parser.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    parser.add_argument("--sets", nargs="+", type=int, default=DECISION_SETS,
                        help="leader sets per policy (num_decision_sets)")
    parser.add_argument("--scales", nargs="+", type=float, default=SCALES)
    parser.add_argument("--traces", nargs="+", default=TRACES)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="simulations running at the same time")
    parser.add_argument("--warmup", type=int, default=25000000)
    parser.add_argument("--sim", type=int, default=25000000)
    parser.add_argument("--dry-run", action="store_true", help="only list what would be built and run")
    args = parser.parse_args()

    # ---- preflight checks ----
    for tool in ("make", "gunzip"):
        if shutil.which(tool) is None:
            sys.exit(f"'{tool}' not found; run this from WSL/Linux")
    for sets in args.sets:
        if sets <= 0 or L2C_SET % sets or L2C_SET // sets < 2:
            sys.exit(f"decision sets must divide {L2C_SET} and leave a stride of at least 2, got {sets}")
    for trace in args.traces:
        if not os.path.exists(os.path.join(ROOT, "Trace_Files", f"{trace}.gz")):
            sys.exit(f"Missing trace Trace_Files/{trace}.gz")

    cpus = os.cpu_count() or 1
    jobs = max(1, min(args.jobs, cpus - 1 if cpus > 2 else cpus))
    free_mb = free_memory_mb()
    if free_mb is not None and free_mb // MIN_FREE_MB_PER_JOB < jobs:
        jobs = max(1, free_mb // MIN_FREE_MB_PER_JOB)
    if jobs != args.jobs:
        print(f"Limiting parallel simulations to {jobs} (requested {args.jobs}, "
              f"{cpus} CPUs, {free_mb} MB available memory)")

    # read every policy file once, so edits during the sweep cannot leak into later builds
    templates, banners = {}, {}
    for policy in args.policy:
        policy_file = os.path.join(ROOT, "replacement", f"{policy}.l2c_repl")
        if not os.path.exists(policy_file):
            sys.exit(f"Missing policy file {policy_file}")
        with open(policy_file, newline="") as f:
            templates[policy] = f.read()
        match = BANNER.search(templates[policy])
        banners[policy] = match.group(1) if match else None

    plan = []  # (policy, name, threshold, sets, scale, source, pending traces)
    for policy in args.policy:
        out_dir = os.path.join(ROOT, "replacement_results", policy)
        for sets in args.sets:
            for scale in args.scales:
                for threshold in args.thresholds:
                    name = config_name(policy, threshold, sets, scale)
                    pending = [t for t in args.traces
                               if not finished(os.path.join(out_dir, f"{name}_{t}.txt"), args.warmup, args.sim)]
                    if pending:
                        source = make_source(templates[policy], threshold, sets, scale)
                        plan.append((policy, name, threshold, sets, scale, source, pending))

    total_runs = sum(len(p[-1]) for p in plan)
    print(f"{len(plan)} configurations to build, {total_runs} simulations to run ({jobs} at a time)")
    if args.dry_run:
        for policy, name, _, _, _, _, pending in plan:
            print(f"  {name}: {' '.join(pending)}")
        return
    if not plan:
        return

    lock = acquire_lock()
    log = os.path.join(ROOT, "replacement_results", "sweep_build.log")
    prepare_tree(log)

    manifest_path = os.path.join(ROOT, MANIFEST)
    new_manifest = not os.path.exists(manifest_path)
    manifest = open(manifest_path, "a")
    if new_manifest:
        manifest.write("config,policy,threshold,decision_sets,scale,source_sha256,binary\n")

    failures = []
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = []
            for policy, name, threshold, sets, scale, source, pending in plan:
                out_dir = os.path.join(ROOT, "replacement_results", policy)
                os.makedirs(out_dir, exist_ok=True)
                binary = build(source, name, log)
                digest = hashlib.sha256(source.encode()).hexdigest()[:16]
                manifest.write(f"{name},{policy},{threshold},{sets},{scale},{digest},{os.path.relpath(binary, ROOT)}\n")
                manifest.flush()
                print(f"built {name}", flush=True)
                for trace in pending:
                    out_path = os.path.join(out_dir, f"{name}_{trace}.txt")
                    futures.append(pool.submit(simulate, binary, trace, out_path, args.warmup, args.sim,
                                               banners[policy], threshold, sets, scale))
            for future in as_completed(futures):
                out_path, error = future.result()
                done += 1
                if error:
                    failures.append((out_path, error))
                    print(f"[{done}/{total_runs}] FAILED {os.path.basename(out_path)}: {error}", flush=True)
                else:
                    print(f"[{done}/{total_runs}] done {os.path.basename(out_path)}", flush=True)
    finally:
        manifest.close()
        # leave the tree the way build_champsim.sh does
        shutil.copyfile(os.path.join(ROOT, "replacement/lru.l2c_repl"), os.path.join(ROOT, L2C_FILE))
        lock.close()

    if failures:
        print(f"\n{len(failures)} simulations failed (re-run the same command to retry them):")
        for out_path, error in failures:
            print(f"  {out_path}: {error}")
        sys.exit(1)
    print("\nAll simulations finished.")


if __name__ == "__main__":
    main()
