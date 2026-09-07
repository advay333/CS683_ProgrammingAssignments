#!/usr/bin/env bash
#
# run_perf_profile_optimized.sh  --  microarchitectural profiling of the final
#                                    optimized kernel, CS683 PA-1 Task 1.
#
# Same driver as run_perf_profile.sh, cut down to the single stage: it builds
# and profiles src/conv_optimized.cpp only, into its own output directory, so
# it can be run without re-collecting the naive / reorder / unroll / tile /
# simd sweeps that are already on disk.
#
# Unlike scripts/run_speedup.sh this driver never touches the Makefile, never
# builds bin/conv, and never patches a source file.  Each src/conv_<stage>.cpp
# is compiled once, standalone, into its own binary whose main() takes the
# kernel variant as argv[5], so a whole sweep runs off one build per stage and
# the counters contain nothing but that one kernel plus its input generation.
#
# The calibration run is NOT included by default: its counters depend only on
# (H, W, K, seed), not on the kernel, so the calibration data already collected
# by run_perf_profile.sh applies unchanged to this sweep.  Pass
# --with-calibration to collect a fresh copy anyway -- worth doing if this sweep
# runs on a different machine, a different compiler, or a different size grid.
#
# Outputs, all under --outdir:
#   build.log                 compiler output for every binary
#   run_manifest.txt          exact configuration this sweep was run with
#   logs/<label>.log          verbatim perf output + program stdout, per run
#   perf_raw_long.csv         one row per (run, event): value, unit, runtime, pct
#   perf_metrics.csv          one row per run, counters pivoted wide
#
# perf_raw_long.csv is the lossless record -- perf_metrics.csv is derived from
# it and can be rebuilt offline with summarize_perf.py.
#
# Usage:
#   ./run_perf_profile.sh [options]
#     -o, --outdir DIR     output directory        (default: ./logs_perf_profile_optimized)
#     -s, --sizes LIST     comma-separated NxN     (default: 64..16384, powers of 2)
#     -r, --reps N         repetitions per point   (default: 20)
#     -k, --kernel N       kernel size K           (default: 3)
#         --seed N         RNG seed                (default: 1234)
#         --cpu ID         taskset core            (default: 0; "off" to not pin)
#         --only REGEX     run only labels matching REGEX
#         --no-sudo        do not prefix perf with sudo
#         --no-build       reuse the binaries already in bin/
#         --with-calibration  also profile src/perf_caliberation.cpp on this grid
#     -n, --dry-run        print the plan, run nothing
#     -h, --help           this message

set -u -o pipefail

# ------------------------------------------------------------------ defaults --
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$ROOT_DIR/src"
INC_DIR="$ROOT_DIR/include"
BIN_DIR="$ROOT_DIR/bin"

OUTDIR="$ROOT_DIR/logs_perf_profile_optimized"
SIZES="64,128,256,512,1024,2048,4096,8192,16384"
REPS=20
KERNEL=3
SEED=1234
PIN_CPU=1
ONLY=""
USE_SUDO=1
DO_BUILD=1
DRY_RUN=0
WITH_CALIB=0

# Compiler flags, identical to the ones pinned in the repository Makefile and to
# the commands recorded in the comment block of each source file.
CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Wall -DSTANDALONE_TEST"

die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
warn() { printf 'warning: %s\n' "$*" >&2; }
info() { printf '%s\n' "$*"; }

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        -o|--outdir)  OUTDIR="$2"; shift 2 ;;
        -s|--sizes)   SIZES="$2"; shift 2 ;;
        -r|--reps)    REPS="$2"; shift 2 ;;
        -k|--kernel)  KERNEL="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --cpu)        PIN_CPU="$2"; shift 2 ;;
        --only)       ONLY="$2"; shift 2 ;;
        --no-sudo)    USE_SUDO=0; shift ;;
        --no-build)   DO_BUILD=0; shift ;;
        --with-calibration) WITH_CALIB=1; shift ;;
        -n|--dry-run) DRY_RUN=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) die "unknown option '$1' (try --help)" ;;
    esac
done

case "$OUTDIR" in /*) : ;; *) OUTDIR="$ROOT_DIR/${OUTDIR#./}" ;; esac

# ---------------------------------------------------------------- the matrix --
# stage | variant passed as argv[5] | label used for filenames and CSV rows
#
# conv_optimized.cpp holds a single implementation, so the variant is "default"
# exactly as for the reorder stage: main() checks it and refuses anything else
# rather than silently profiling the wrong thing.  To compare several optimized
# variants later, give them names inside the file, dispatch on argv[5] the way
# conv_simd.cpp does, and add one line here per variant.
RUNS=( "optimized|default|optimized" )

# The calibration binary is opt-in here (see the header comment).
if [ "$WITH_CALIB" -eq 1 ]; then
    RUNS=( "calibration|default|calibration" "${RUNS[@]}" )
fi

# stage -> source file / binary
src_for() {
    case "$1" in
        calibration) printf '%s' "$SRC_DIR/perf_caliberation.cpp" ;;
        *)           printf '%s' "$SRC_DIR/conv_$1.cpp" ;;
    esac
}
bin_for() {
    case "$1" in
        calibration) printf '%s' "$BIN_DIR/perf_caliberation" ;;
        *)           printf '%s' "$BIN_DIR/conv_prof_$1" ;;
    esac
}

# ------------------------------------------------------------------- events --
# On a hybrid Intel part the core PMU events must be qualified with cpu_core/,
# or perf either refuses them or silently counts on one core type only.  The
# software events (context-switches, page-faults) are never qualified.
if [ -d /sys/bus/event_source/devices/cpu_core ]; then
    PMU="cpu_core/"; PMU_END="/"
    info "hybrid PMU detected: qualifying core events with cpu_core/"
else
    PMU=""; PMU_END=""
fi

CORE_EVENTS=(
    L1-dcache-loads
    L1-dcache-load-misses
    branch-instructions
    branch-misses
    instructions
    cpu-cycles
)
SW_EVENTS=( context-switches page-faults )

EVENTS=""
for e in "${CORE_EVENTS[@]}"; do EVENTS="${EVENTS}${EVENTS:+,}${PMU}${e}${PMU_END}"; done
for e in "${SW_EVENTS[@]}";   do EVENTS="${EVENTS}${EVENTS:+,}${e}"; done

# Four programmable counters plus two fixed ones: this fits on a P-core without
# multiplexing.  The parser flags any run where perf reports <100% enabled time
# anyway, since that silently turns every count into an extrapolation.

# ------------------------------------------------------------------ prefixes --
SUDO=()
[ "$USE_SUDO" -eq 1 ] && SUDO=(sudo)

PIN=()
if [ "$PIN_CPU" != "off" ] && [ "$PIN_CPU" != "none" ] && [ -n "$PIN_CPU" ]; then
    if command -v taskset >/dev/null 2>&1; then
        PIN=(taskset -c "$PIN_CPU")
    else
        warn "taskset not found; running unpinned"
    fi
fi

# ------------------------------------------------------------------- planning --
IFS=',' read -r -a SIZE_LIST <<< "$SIZES"
SELECTED=()
for spec in "${RUNS[@]}"; do
    label="${spec##*|}"
    if [ -n "$ONLY" ] && ! printf '%s' "$label" | grep -Eq "$ONLY"; then continue; fi
    SELECTED+=("$spec")
done
[ ${#SELECTED[@]} -gt 0 ] || die "no runs match --only '$ONLY'"

TOTAL=$(( ${#SELECTED[@]} * ${#SIZE_LIST[@]} * REPS ))

info ""
info "=============================================================="
info " runs      : ${#SELECTED[@]}   ($(printf '%s ' "${SELECTED[@]##*|}"))"
info " sizes     : $SIZES"
info " K=$KERNEL  seed=$SEED  reps=$REPS  cpu=${PIN_CPU}"
info " events    : $EVENTS"
info " outdir    : $OUTDIR"
info " total     : $TOTAL profiled executions"
info "=============================================================="

# The largest workload allocates img + out + padded input.  At 16384x16384 that
# is roughly 3 GiB resident; check before spending an hour to hit the OOM killer.
BIGGEST=0
for s in "${SIZE_LIST[@]}"; do [ "$s" -gt "$BIGGEST" ] && BIGGEST="$s"; done
NEED_MB=$(( BIGGEST * BIGGEST * 4 * 3 / 1048576 ))
AVAIL_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$AVAIL_MB" -gt 0 ] && [ "$NEED_MB" -gt "$AVAIL_MB" ]; then
    warn "largest workload needs about ${NEED_MB} MiB but only ${AVAIL_MB} MiB is available"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    for spec in "${SELECTED[@]}"; do
        IFS='|' read -r stage variant label <<< "$spec"
        for s in "${SIZE_LIST[@]}"; do
            info "  would run x$REPS : $(bin_for "$stage") $s $s $KERNEL $SEED $variant   [$label]"
        done
    done
    exit 0
fi

# --------------------------------------------------------------------- build --
mkdir -p "$OUTDIR/logs" "$BIN_DIR" || die "cannot create $OUTDIR"
BUILD_LOG="$OUTDIR/build.log"
: > "$BUILD_LOG"

if [ "$DO_BUILD" -eq 1 ]; then
    info ""
    info "building standalone profiling binaries ..."
    seen_stage=" "
    for spec in "${SELECTED[@]}"; do
        IFS='|' read -r stage variant label <<< "$spec"
        case "$seen_stage" in *" $stage "*) continue ;; esac
        seen_stage="$seen_stage$stage "

        src="$(src_for "$stage")"; bin="$(bin_for "$stage")"
        [ -f "$src" ] || die "missing source: $src"
        info "  $stage  ->  $bin"
        {
            echo "### $stage"
            echo "\$ $CXX $CXXFLAGS -I$INC_DIR $src -o $bin"
        } >> "$BUILD_LOG"
        # shellcheck disable=SC2086
        if ! $CXX $CXXFLAGS -I"$INC_DIR" "$src" -o "$bin" >> "$BUILD_LOG" 2>&1; then
            tail -40 "$BUILD_LOG" >&2
            die "build failed for $stage (see $BUILD_LOG)"
        fi
    done
    info "build ok"
fi

for spec in "${SELECTED[@]}"; do
    IFS='|' read -r stage variant label <<< "$spec"
    [ -x "$(bin_for "$stage")" ] || die "missing binary $(bin_for "$stage") (drop --no-build)"
done

# Ask for the sudo credential once, so the sweep is not interrupted by a prompt
# in the middle of a measurement.
if [ "$USE_SUDO" -eq 1 ]; then
    sudo -v || die "sudo required for perf (or pass --no-sudo if perf_event_paranoid allows it)"
fi

# ---------------------------------------------------------------- CSV headers --
LONG_CSV="$OUTDIR/perf_raw_long.csv"
WIDE_CSV="$OUTDIR/perf_metrics.csv"
printf 'label,stage,variant,H,W,K,seed,rep,event,value,unit,counter_runtime_ns,pct_enabled\n' > "$LONG_CSV"
printf 'label,stage,variant,H,W,K,seed,rep,exit_status,min_pct_enabled,%s\n' \
    "$(IFS=,; printf '%s' "${CORE_EVENTS[*]},${SW_EVENTS[*]}")" > "$WIDE_CSV"

{
    echo "run_perf_profile.sh manifest"
    echo "started      : $(date -Is)"
    echo "host         : $(hostname 2>/dev/null || echo unknown)"
    echo "kernel       : $(uname -srm)"
    echo "cpu          : $(awk -F: '/model name/ {print $2; exit}' /proc/cpuinfo 2>/dev/null | sed 's/^ *//')"
    echo "compiler     : $($CXX --version | head -1)"
    echo "cxxflags     : $CXXFLAGS"
    echo "perf         : $(perf --version 2>/dev/null || echo 'not found')"
    echo "paranoid     : $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo '?')"
    echo "governor     : $(cat /sys/devices/system/cpu/cpu${PIN_CPU}/cpufreq/scaling_governor 2>/dev/null || echo '?')"
    echo "turbo(no_turbo): $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')"
    echo "smt          : $(cat /sys/devices/system/cpu/smt/active 2>/dev/null || echo '?')"
    echo "events       : $EVENTS"
    echo "sizes        : $SIZES"
    echo "K            : $KERNEL"
    echo "seed         : $SEED"
    echo "reps         : $REPS"
    echo "pinned cpu   : $PIN_CPU"
    echo "runs         : ${SELECTED[*]}"
} > "$OUTDIR/run_manifest.txt"

TMP_PERF="$(mktemp)"
trap 'rm -f "$TMP_PERF"' EXIT

# ---------------------------------------------------------------- the sweep --
DONE=0
STARTED_AT=$(date +%s)

for spec in "${SELECTED[@]}"; do
    IFS='|' read -r stage variant label <<< "$spec"
    bin="$(bin_for "$stage")"
    RUN_LOG="$OUTDIR/logs/${label}.log"

    {
        echo "############################################################"
        echo "# label   : $label"
        echo "# stage   : $stage      ($(src_for "$stage"))"
        echo "# variant : $variant"
        echo "# events  : $EVENTS"
        echo "# started : $(date -Is)"
        echo "############################################################"
    } > "$RUN_LOG"

    info ""
    info "--- $label ---"

    for size in "${SIZE_LIST[@]}"; do
        H="$size"; W="$size"
        for ((rep = 1; rep <= REPS; rep++)); do

            # perf writes its counters to stderr; redirecting stderr here (rather
            # than using perf -o) keeps the file owned by the invoking user even
            # though perf itself runs under sudo.
            prog_out="$( "${SUDO[@]}" perf stat -x, -e "$EVENTS" \
                            "${PIN[@]}" "$bin" "$H" "$W" "$KERNEL" "$SEED" "$variant" \
                            2> "$TMP_PERF" )"
            rc=$?

            {
                echo ""
                echo "----- [$label] H=$H W=$W K=$KERNEL seed=$SEED rep=$rep/$REPS exit=$rc -----"
                printf '%s\n' "$prog_out"
                echo "--- perf stat -x, ---"
                cat "$TMP_PERF"
            } >> "$RUN_LOG"

            if [ $rc -ne 0 ]; then
                warn "$label H=$H rep=$rep exited with status $rc (see $RUN_LOG)"
            fi

            # Append the long-format rows and one pivoted row for this execution.
            awk -F',' -v OFS=',' \
                -v label="$label" -v stage="$stage" -v variant="$variant" \
                -v H="$H" -v W="$W" -v K="$KERNEL" -v seed="$SEED" -v rep="$rep" \
                -v rc="$rc" -v longf="$LONG_CSV" -v widef="$WIDE_CSV" \
                -v order="$(IFS=,; printf '%s' "${CORE_EVENTS[*]},${SW_EVENTS[*]}")" '
                function strip(e) { sub(/^cpu_core\//, "", e); sub(/\/$/, "", e); return e }
                # perf -x, rows: value,unit,event,counter-runtime,pct-enabled[,...]
                NF >= 5 && $3 != "" {
                    ev  = strip($3)
                    val = $1
                    if (val ~ /^</) { val = "" }            # <not counted> / <not supported>
                    pct = ($5 == "" ? "" : $5 + 0)
                    print label, stage, variant, H, W, K, seed, rep, ev, val, $2, $4, pct >> longf
                    v[ev] = val
                    if (pct != "" && (minpct == "" || pct < minpct)) minpct = pct
                }
                END {
                    n = split(order, ord, ",")
                    line = label OFS stage OFS variant OFS H OFS W OFS K OFS seed OFS rep OFS rc OFS minpct
                    for (i = 1; i <= n; i++) line = line OFS (ord[i] in v ? v[ord[i]] : "")
                    print line >> widef
                    if (minpct != "" && minpct < 99.5)
                        printf("warning: %s H=%s rep=%s counted only %.1f%% of the time (event multiplexing)\n",
                               label, H, rep, minpct) > "/dev/stderr"
                }' "$TMP_PERF"

            DONE=$((DONE + 1))
        done

        # Progress with a rough ETA, since a full sweep is measured in hours.
        ELAPSED=$(( $(date +%s) - STARTED_AT ))
        if [ "$DONE" -gt 0 ]; then
            ETA=$(( ELAPSED * (TOTAL - DONE) / DONE ))
            printf '  %-14s %6s  [%d/%d done, elapsed %dm, eta ~%dm]\n' \
                "$label" "${H}x${W}" "$DONE" "$TOTAL" $((ELAPSED / 60)) $((ETA / 60))
        fi
    done

    echo "# finished : $(date -Is)" >> "$RUN_LOG"
done

echo "finished     : $(date -Is)" >> "$OUTDIR/run_manifest.txt"

info ""
info "=============================================================="
info " sweep complete: $DONE executions"
info " raw logs   : $OUTDIR/logs/<label>.log"
info " long CSV   : $LONG_CSV"
info " wide CSV   : $WIDE_CSV"
info " manifest   : $OUTDIR/run_manifest.txt"
info "=============================================================="

if command -v python3 >/dev/null 2>&1 && [ -f "$ROOT_DIR/summarize_perf.py" ]; then
    info ""
    python3 "$ROOT_DIR/summarize_perf.py" "$WIDE_CSV" --outdir "$OUTDIR"
else
    info ""
    info "run summarize_perf.py on $WIDE_CSV to get per-point medians and derived metrics"
fi