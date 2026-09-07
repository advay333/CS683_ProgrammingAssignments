#!/usr/bin/env bash
# sweep_prefetch.sh  a_degree / b_degree / hardware-prefetcher sweep for the three
# software-prefetch variants in src/matmul_prefetch.cpp.
#
# Run from the repository root (the directory containing Makefile, src/ and include/):
#
#     ./sweep_prefetch.sh
#
# Grid: 3 variants x 4 a_degree x 4 b_degree x 2 hardware-prefetcher states = 96 configs.
# For each one it patches the constants inside that ONE function, rebuilds with the pinned
# Makefile flags, sets the core-0 prefetcher MSR, and times the kernel on every matrix size.
# Raw terminal output -> results/logs/<variant>_a<A>_b<B>_hw<on|off>.txt (96 files).
# Every timed sample -> one row in results/results.csv. No averaging is done anywhere.
#
# HARDWARE PREFETCHER CONTROL
#   MSR 0x1A4 on core 0: value 15 (0xF) disables all four prefetchers, 0 enables them.
#   This is per-physical-core, so the workload MUST run on the core whose MSR was written.
#   The script therefore pins to CPU $MSR_CPU (default 0) with taskset and refuses to run
#   the hw dimension without it. The MSR is restored to 0 (all prefetchers on) on exit,
#   including on Ctrl-C, and is read back with rdmsr after every write.
#
#   Requirements: msr-tools (wrmsr/rdmsr), the `msr` kernel module, and sudo. Run
#   `sudo -v` first, or the sweep will stop on a password prompt partway through.
#
# src/matmul_prefetch.cpp is backed up on start and restored on exit.
#
# Knobs (override from the environment, e.g. `SAMPLES=3 ./sweep_prefetch.sh`):
#   VARIANTS A_DEGREES B_DEGREES HW_STATES SIZES SAMPLES SEED BASELINE_SAMPLES
#   VERIFY_MAX_N RESUME MSR_CPU SKIP_MSR OUTDIR

set -uo pipefail

# ------------------------------- configuration ------------------------------
VARIANTS=${VARIANTS:-"v1 tiled tiled_simd"}
A_DEGREES=${A_DEGREES:-"1 3 5 7"}
B_DEGREES=${B_DEGREES:-"16 48 80 128"}
HW_STATES=${HW_STATES:-"on off"}          # hardware prefetcher: on = MSR 0, off = MSR 15
SIZES=${SIZES:-"256 512 1024 2048"}       # square: M = N = K
SAMPLES=${SAMPLES:-5}                     # independent timed runs per data point
SEED=${SEED:-1234}
BASELINE_SAMPLES=${BASELINE_SAMPLES:-3}   # naive runs per (size, hw state)

# Correctness needs a naive reference pass (~17 s at 2048), so sizes above this are timed
# but not verified. Set VERIFY_MAX_N=2048 to verify everything.
VERIFY_MAX_N=${VERIFY_MAX_N:-512}

RESUME=${RESUME:-1}                       # 1 = skip configs whose log is already complete
MSR_CPU=${MSR_CPU:-0}                     # core whose MSR is written AND on which we run
SKIP_MSR=${SKIP_MSR:-0}                   # 1 = don't touch the MSR (dry run of the harness)
OUTDIR=${OUTDIR:-results}

MSR_PREFETCH=0x1a4
MSR_OFF_VAL=15                            # all four prefetchers disabled
MSR_ON_VAL=0                              # all four prefetchers enabled

SRC=src/matmul_prefetch.cpp
PATCHER=tools/patch_degrees.py
BIN=bin/matmul_profile
LOGDIR="$OUTDIR/logs"
CSV="$OUTDIR/results.csv"
BASEFILE="$OUTDIR/baselines.txt"

# ------------------------------- sanity checks ------------------------------
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -f Makefile ]]  || die "no Makefile here - run this from the repository root."
[[ -f "$SRC" ]]    || die "$SRC not found."
[[ -f src/profile_main.cpp ]] || die "src/profile_main.cpp not found - copy it in first."
[[ -f "$PATCHER" ]] || die "$PATCHER not found - copy it in first."
command -v python3 >/dev/null || die "python3 is required for the patcher."

mkdir -p "$LOGDIR"

# ---- MSR preflight ---------------------------------------------------------
# Pinning is mandatory: MSR 0x1A4 is per-core, so timing a run on a different core than the
# one we wrote would silently mix the two prefetcher states.
if [[ "$SKIP_MSR" != "1" ]]; then
    command -v taskset >/dev/null || \
        die "taskset not found; required to pin to core $MSR_CPU. Set SKIP_MSR=1 and HW_STATES=on to sweep without the hardware dimension."
    command -v wrmsr >/dev/null || die "wrmsr not found - install msr-tools."
    command -v rdmsr >/dev/null || die "rdmsr not found - install msr-tools."
    sudo modprobe msr 2>/dev/null || \
        printf 'note: modprobe msr failed (may be built in); continuing.\n' >&2
    sudo -v || die "sudo credentials are required for wrmsr."
    MSR_ORIGINAL=$(sudo rdmsr -p "$MSR_CPU" "$MSR_PREFETCH" 2>/dev/null) || \
        die "cannot read MSR $MSR_PREFETCH on cpu $MSR_CPU."
    printf 'MSR %s on cpu %s currently reads 0x%s (prefetcher re-enabled on exit).\n' \
        "$MSR_PREFETCH" "$MSR_CPU" "$MSR_ORIGINAL" >&2
else
    MSR_ORIGINAL=""
    printf 'SKIP_MSR=1: the hardware prefetcher will NOT be touched.\n' >&2
fi

RUNNER=()
if command -v taskset >/dev/null; then
    RUNNER=(taskset -c "$MSR_CPU")
fi

# ---- restore the source and the MSR no matter how we exit ----
BACKUP="$SRC.sweep_backup.$$"
cp "$SRC" "$BACKUP" || die "could not back up $SRC"
cleanup() {
    if [[ -f "$BACKUP" ]]; then
        cp "$BACKUP" "$SRC" && rm -f "$BACKUP"
        printf '\n%s restored from backup.\n' "$SRC" >&2
    fi
    if [[ "$SKIP_MSR" != "1" ]]; then
        if sudo wrmsr -p "$MSR_CPU" "$MSR_PREFETCH" "$MSR_ON_VAL" 2>/dev/null; then
            printf 'hardware prefetcher re-enabled on cpu %s (MSR <- %s).\n' \
                "$MSR_CPU" "$MSR_ON_VAL" >&2
        else
            printf 'WARNING: failed to re-enable the prefetcher on cpu %s. Run:\n  sudo wrmsr -p %s %s %s\n' \
                "$MSR_CPU" "$MSR_CPU" "$MSR_PREFETCH" "$MSR_ON_VAL" >&2
        fi
    fi
}
trap cleanup EXIT INT TERM

# ---- set + verify the prefetcher state; echoes what the MSR reads back ----
set_prefetcher() {
    local state=$1 want got
    [[ "$state" == "off" ]] && want=$MSR_OFF_VAL || want=$MSR_ON_VAL
    if [[ "$SKIP_MSR" == "1" ]]; then
        printf 'hw prefetch : %s (SKIP_MSR=1, MSR untouched)\n' "$state"
        return 0
    fi
    if ! sudo -n wrmsr -p "$MSR_CPU" "$MSR_PREFETCH" "$want" 2>/dev/null; then
        sudo -v || return 1
        sudo wrmsr -p "$MSR_CPU" "$MSR_PREFETCH" "$want" || return 1
    fi
    got=$(sudo rdmsr -p "$MSR_CPU" "$MSR_PREFETCH" 2>/dev/null)
    # rdmsr prints hex with no 0x prefix; compare numerically.
    if [[ -z "$got" ]] || (( 16#$got != want )); then
        printf 'hw prefetch : MSR READ-BACK MISMATCH (wanted %d, got 0x%s)\n' \
            "$want" "${got:-?}"
        return 1
    fi
    printf 'hw prefetch : %s  (MSR %s on cpu %s = 0x%s, verified)\n' \
        "$state" "$MSR_PREFETCH" "$MSR_CPU" "$got"
    return 0
}

# ---- add the profile target to the Makefile once, with the same pinned flags ----
if ! grep -q 'matmul_profile' Makefile; then
    cat >> Makefile <<'EOF'

# --- added by sweep_prefetch.sh: profiling driver, same CXXFLAGS as bin/matmul ---
PROFILE_MAIN := src/profile_main.cpp

.PHONY: profile
profile: $(BIN)/matmul_profile
$(BIN)/matmul_profile: $(STUDENT_SRC) $(PROFILE_MAIN) | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
EOF
    printf 'Makefile: appended the `profile` target.\n' >&2
fi

build() {
    make profile >/dev/null 2>"$OUTDIR/last_build_errors.txt"
    local rc=$?
    if (( rc != 0 )); then
        printf 'BUILD FAILED - see %s/last_build_errors.txt\n' "$OUTDIR" >&2
        sed 's/^/  /' "$OUTDIR/last_build_errors.txt" >&2
        return 1
    fi
    return 0
}

# ------------------------------- naive baselines ----------------------------
# The naive kernel does not depend on a_degree/b_degree, so it is timed once per
# (size, hw state) instead of once per config. It DOES depend on the hardware prefetcher,
# hence one baseline set per state: speedups are always against the naive time measured
# under the SAME prefetcher configuration.
printf 'building baseline binary...\n' >&2
build || die "initial build failed."

declare -A BASELINE
{ echo "hw_prefetch size baseline_ms"; } > "$BASEFILE"
for hw in $HW_STATES; do
    set_prefetcher "$hw" >&2 || die "could not set the prefetcher to '$hw'."
    for n in $SIZES; do
        printf 'baseline naive hw=%-3s %dx%dx%d ... ' "$hw" "$n" "$n" "$n" >&2
        out=$("${RUNNER[@]}" ./"$BIN" naive "$n" "$n" "$n" "$SEED" "$BASELINE_SAMPLES" 2>&1)
        ms=$(printf '%s\n' "$out" | awk '/^BASELINE_MS/{print $2}')
        [[ -n "$ms" ]] || die "could not read baseline for hw=$hw N=$n. Output was: $out"
        BASELINE["$hw,$n"]=$ms
        printf '%s %s %s\n' "$hw" "$n" "$ms" >> "$BASEFILE"
        printf '%s ms\n' "$ms" >&2
    done
done

# ------------------------------- CSV header ---------------------------------
if [[ ! -f "$CSV" ]]; then
    echo "variant,a_degree,b_degree,hw_prefetch,M,N,K,seed,sample,time_ms,gflops,baseline_ms,speedup,correct" > "$CSV"
fi

# ------------------------------- the sweep ----------------------------------
total=0
for v in $VARIANTS; do for a in $A_DEGREES; do for b in $B_DEGREES; do
    for h in $HW_STATES; do total=$((total+1)); done; done; done; done
done_n=0
failed=0
start_ts=$(date +%s)

# hw is the OUTERMOST loop so the MSR is written once per state block, not once per config.
for hw in $HW_STATES; do
    set_prefetcher "$hw" >&2 || die "could not set the prefetcher to '$hw'."

    for variant in $VARIANTS; do
        func="matmul_prefetch_${variant}"

        for a in $A_DEGREES; do
            for b in $B_DEGREES; do
                done_n=$((done_n+1))
                log="$LOGDIR/${variant}_a${a}_b${b}_hw${hw}.txt"

                if [[ "$RESUME" == "1" && -s "$log" ]] && \
                   grep -q '^SWEEP_CONFIG_COMPLETE' "$log"; then
                    printf '[%3d/%3d] hw=%-3s %-11s a=%d b=%-3d  (skipped, log complete)\n' \
                        "$done_n" "$total" "$hw" "$variant" "$a" "$b" >&2
                    continue
                fi

                elapsed=$(( $(date +%s) - start_ts ))
                printf '[%3d/%3d] hw=%-3s %-11s a=%d b=%-3d  (%dm%02ds elapsed) ' \
                    "$done_n" "$total" "$hw" "$variant" "$a" "$b" \
                    $((elapsed/60)) $((elapsed%60)) >&2

                # --- patch ONLY this variant, from a pristine source every time
                cp "$BACKUP" "$SRC"
                {
                    echo "=============================================================="
                    echo "variant     : $variant   (function $func)"
                    echo "a_degree    : $a"
                    echo "b_degree    : $b"
                    echo "hw request  : $hw"
                    echo "seed        : $SEED"
                    echo "samples     : $SAMPLES per size (no averaging - see results.csv)"
                    echo "sizes       : $SIZES"
                    echo "pinned to   : cpu $MSR_CPU"
                    echo "host        : $(hostname 2>/dev/null || echo unknown)"
                    echo "date        : $(date -Is)"
                    echo "compiled declarations actually in $SRC:"
                } > "$log"

                if ! python3 "$PATCHER" "$SRC" "$func" "$a" "$b" >> "$log" 2>&1; then
                    printf 'PATCH FAILED\n' >&2
                    echo "SWEEP_CONFIG_FAILED patch" >> "$log"
                    failed=$((failed+1)); continue
                fi

                if ! build; then
                    echo "SWEEP_CONFIG_FAILED build" >> "$log"
                    failed=$((failed+1)); continue
                fi

                # Re-assert and re-verify the MSR for every config, so each log carries
                # first-hand evidence of the prefetcher state its numbers were taken under.
                if ! set_prefetcher "$hw" >> "$log" 2>&1; then
                    printf 'MSR SET FAILED\n' >&2
                    echo "SWEEP_CONFIG_FAILED msr" >> "$log"
                    failed=$((failed+1)); continue
                fi

                # The binary md5 differs per config: proof that the patched source really
                # was recompiled and that this run used the edited kernel, not a stale build.
                {
                    echo "build       : ok"
                    echo "binary md5  : $(md5sum "$BIN" | cut -d' ' -f1)"
                    echo "=============================================================="
                } >> "$log"

                for n in $SIZES; do
                    verify=1
                    (( n > VERIFY_MAX_N )) && verify=0
                    {
                        echo
                        echo "--- M=N=K=$n  (verify=$verify, hw=$hw, naive baseline ${BASELINE[$hw,$n]} ms) ---"
                    } >> "$log"

                    out=$("${RUNNER[@]}" ./"$BIN" "$variant" "$n" "$n" "$n" "$SEED" \
                            "$SAMPLES" "${BASELINE[$hw,$n]}" "$verify" 2>&1)
                    printf '%s\n' "$out" >> "$log"

                    # CSV,variant,M,... -> variant,a,b,hw,M,...
                    printf '%s\n' "$out" | grep '^CSV,' \
                        | sed -E "s/^CSV,([^,]*),/\1,${a},${b},${hw},/" >> "$CSV"

                    if printf '%s\n' "$out" | grep -q 'correct=NO'; then
                        printf 'INCORRECT@%d ' "$n" >&2
                    fi
                done

                echo "SWEEP_CONFIG_COMPLETE" >> "$log"
                printf 'ok\n' >&2
            done
        done
    done
done

elapsed=$(( $(date +%s) - start_ts ))
printf '\nsweep finished in %dm%02ds: %d configs, %d failed.\n' \
    $((elapsed/60)) $((elapsed%60)) "$total" "$failed" >&2
printf 'logs : %s/\nrows : %s (%d data rows)\n' \
    "$LOGDIR" "$CSV" "$(( $(wc -l < "$CSV") - 1 ))" >&2