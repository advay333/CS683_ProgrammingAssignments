#!/usr/bin/env bash
#
# run_speedup.sh  automated speedup-analysis driver for CS683 PA-1 Task 1.
#
# Reads a config file describing WHICH stage files, WHICH version of the kernel
# inside each file, WHICH matrix sizes and WHICH kernel sizes to benchmark, then
# for every (stage, version) pair:
#
#   1. stages a private copy of Makefile + include/ + src/ under <outdir>/.build/,
#   2. rewrites ONLY the delegating call inside the master dispatcher
#      (e.g. conv_simd(...) -> calls conv_simd256_v1(...)) in that copy,
#   3. runs "make clean && make all" in the copy (the repo Makefile is copied
#      verbatim, so the pinned -O2 -fno-tree-vectorize -mavx2 -mfma flags are used),
#   4. runs ./bin/conv <stage> H W K seed for every requested workload,
#   5. appends the raw harness output to <outdir>/<stage>_<label>_speedup.txt.
#
# The repository's own Makefile and src/*.cpp are NEVER modified: all patching
# happens on the staged copies inside <outdir>/.build/.
#
# Usage:
#   scripts/run_speedup.sh [-c config.txt] [-o outdir] [options]
#
#   -c, --config FILE    config file (default: scripts/config.txt)
#   -o, --outdir DIR     output directory (overrides "outdir" in the config)
#   -l, --list STAGE     list the kernel versions found in src/conv_<STAGE>.cpp and exit
#   -n, --dry-run        print what would be built/run, build and run nothing
#   -k, --keep-build     keep <outdir>/.build/ after finishing (default)
#       --clean-build    delete <outdir>/.build/ when done
#   -h, --help           this message
#
set -u -o pipefail

# ---------------------------------------------------------------- locations --
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"     # task1/
SRC_DIR="$ROOT_DIR/src"
MAKE="${MAKE:-make}"          # e.g. MAKE=mingw32-make scripts/run_speedup.sh

CONFIG="$SCRIPT_DIR/config.txt"
OUTDIR_CLI=""
DRY_RUN=0
KEEP_BUILD=1
LIST_STAGE=""

die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
warn() { printf 'warning: %s\n' "$*" >&2; }
info() { printf '%s\n' "$*"; }

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# ------------------------------------------------------------------- args ----
while [ $# -gt 0 ]; do
    case "$1" in
        -c|--config)  CONFIG="$2"; shift 2 ;;
        -o|--outdir)  OUTDIR_CLI="$2"; shift 2 ;;
        -l|--list)    LIST_STAGE="$2"; shift 2 ;;
        -n|--dry-run) DRY_RUN=1; shift ;;
        -k|--keep-build)  KEEP_BUILD=1; shift ;;
        --clean-build)    KEEP_BUILD=0; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) die "unknown option '$1' (try --help)" ;;
    esac
done

# ------------------------------------------------------- version discovery ---
# Every free function in src/conv_<stage>.cpp whose name starts with conv_ is a
# candidate version; the master dispatcher itself is reported as "default".
list_versions() {
    local stage="$1"
    local file="$SRC_DIR/conv_$stage.cpp"
    [ -f "$file" ] || die "no such stage file: $file"
    grep -oE '^[[:space:]]*void[[:space:]]+conv_[A-Za-z0-9_]+[[:space:]]*\(' "$file" \
        | grep -oE 'conv_[A-Za-z0-9_]+' \
        | grep -vx "conv_$stage"
}

if [ -n "$LIST_STAGE" ]; then
    info "versions available in src/conv_$LIST_STAGE.cpp:"
    info "  default        (whatever conv_$LIST_STAGE currently dispatches to)"
    list_versions "$LIST_STAGE" | sed 's/^/  /'
    exit 0
fi

# ------------------------------------------------------------ config parse ---
[ -r "$CONFIG" ] || die "config file not found (or not readable): $CONFIG"

# globals with defaults
G_OUTDIR="results"
G_SEED="1234"
G_SIZES="1024x1024"
G_KERNELS="3"
G_REPEAT="1"
G_PIN_CPU=""

RUN_SPECS=()     # one entry per "run" line, kept verbatim for later parsing

trim() { printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }

lineno=0
while IFS= read -r raw || [ -n "$raw" ]; do
    lineno=$((lineno + 1))
    line="${raw%%#*}"                       # strip comments
    line="$(trim "$line")"
    [ -z "$line" ] && continue

    if [ "${line:0:4}" = "run " ]; then
        RUN_SPECS+=("$(trim "${line#run }")")
        continue
    fi

    case "$line" in
        *=*) : ;;
        *) die "$CONFIG:$lineno: cannot parse '$line' (expected 'key = value' or 'run <stage> : <version>')" ;;
    esac
    key="$(trim "${line%%=*}")"
    val="$(trim "${line#*=}")"
    case "$key" in
        outdir)  G_OUTDIR="$val" ;;
        seed)    G_SEED="$val" ;;
        sizes)   G_SIZES="$val" ;;
        kernels) G_KERNELS="$val" ;;
        repeat)  G_REPEAT="$val" ;;
        pin_cpu) G_PIN_CPU="$val" ;;
        *) die "$CONFIG:$lineno: unknown key '$key'" ;;
    esac
done < "$CONFIG"

[ ${#RUN_SPECS[@]} -gt 0 ] || die "$CONFIG: no 'run <stage> : <version>' lines found"

OUTDIR="${OUTDIR_CLI:-$G_OUTDIR}"
case "$OUTDIR" in /*|?:[\\/]*) : ;; *) OUTDIR="$ROOT_DIR/$OUTDIR" ;; esac
BUILD_ROOT="$OUTDIR/.build"

# ------------------------------------------------------------- run helpers ---
# csv list "a, b ,c" -> newline separated
split_list() {
    printf '%s' "$1" | tr ',' '\n' \
        | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | grep -v '^$'
}

# default label for a version: conv_unroll_v3 -> v3, conv_simd256_v3 -> simd256_v3
default_label() {
    local stage="$1"
    local version="$2"
    local lbl
    if [ "$version" = "default" ]; then printf 'default'; return; fi
    lbl="${version#conv_}"
    lbl="${lbl#${stage}_}"
    printf '%s' "$lbl"
}

# stage a patched copy of the tree and build it; echoes the staged dir
stage_and_build() {
    local stage="$1"
    local version="$2"
    local label="$3"
    local dir="$BUILD_ROOT/${stage}_${label}"

    rm -rf "$dir"
    mkdir -p "$dir"
    cp "$ROOT_DIR/Makefile" "$dir/Makefile"     # verbatim copy, never edited
    cp -r "$ROOT_DIR/include" "$dir/include"
    cp -r "$SRC_DIR" "$dir/src"

    if [ "$version" != "default" ]; then
        local target="$dir/src/conv_$stage.cpp"
        grep -qE "^[[:space:]]*void[[:space:]]+$version[[:space:]]*\(" "$target" \
            || die "version '$version' not found in src/conv_$stage.cpp (try --list $stage)"

        # Replace the delegating call inside the body of the master dispatcher
        # void conv_<stage>(...) with a call to the requested version.  Only an
        # uncommented "conv_xxx(in, out, ker, H, W, K);" statement is rewritten,
        # and exactly one such statement must exist or we bail out.
        awk -v master="conv_$stage" -v target="$version" '
            BEGIN { in_master = 0; depth = 0; seen = 0; patched = 0 }
            {
                if (!in_master && $0 ~ ("^[[:space:]]*void[[:space:]]+" master "[[:space:]]*\\(")) {
                    in_master = 1; depth = 0; seen = 0
                }
                if (in_master) {
                    if ($0 ~ /^[[:space:]]*conv_[A-Za-z0-9_]+[[:space:]]*\([[:space:]]*in[[:space:]]*,/) {
                        sub(/conv_[A-Za-z0-9_]+[[:space:]]*\(/, target "(")
                        patched++
                    }
                    n = gsub(/\{/, "{"); m = gsub(/\}/, "}")
                    depth += n - m
                    if (n > 0) seen = 1
                    if (seen && depth <= 0) in_master = 0
                }
                print
            }
            END { if (patched != 1) exit 3 }
        ' "$target" > "$target.new" \
            || die "could not patch the conv_$stage dispatcher for '$version': expected exactly one uncommented 'conv_xxx(in, out, ker, H, W, K);' call inside void conv_$stage(...)"
        mv "$target.new" "$target"
    fi

    ( cd "$dir" && { "$MAKE" clean >/dev/null 2>&1 || true; } && "$MAKE" all ) >"$dir/build.log" 2>&1 \
        || { cat "$dir/build.log" >&2; die "build failed for $stage/$label (see $dir/build.log)"; }

    printf '%s' "$dir"
}

# taskset prefix, if requested and available
PIN_PREFIX=()
if [ -n "$G_PIN_CPU" ] && [ "$G_PIN_CPU" != "off" ] && [ "$G_PIN_CPU" != "none" ]; then
    if command -v taskset >/dev/null 2>&1; then
        PIN_PREFIX=(taskset -c "$G_PIN_CPU")
    else
        warn "pin_cpu=$G_PIN_CPU requested but 'taskset' is not available; running unpinned"
    fi
fi

# ------------------------------------------------------------------- main ----
mkdir -p "$OUTDIR" "$BUILD_ROOT"
CSV="$OUTDIR/summary.csv"
if [ "$DRY_RUN" -eq 0 ]; then
    printf 'stage,label,version,H,W,K,seed,rep,correct,naive_ms,stage_ms,gflops,speedup\n' > "$CSV"
fi

declare -A SEEN_OUT=()
HOST="$(hostname 2>/dev/null || echo unknown)"
STARTED="$(date '+%Y-%m-%d %H:%M:%S')"

# Validate every run line up front, so a typo in the last one does not surface
# only after the earlier sweeps have already spent minutes benchmarking.
for spec in "${RUN_SPECS[@]}"; do
    v_body="${spec%%|*}"
    case "$v_body" in *:*) : ;; *) die "bad run line: 'run $spec' (need '<stage> : <version>')" ;; esac
    v_stage="$(trim "${v_body%%:*}")"
    v_part="$(trim "${v_body#*:}")"
    v_label=""
    case "$v_part" in
        *" as "*) v_label="$(trim "${v_part##* as }")"; v_version="$(trim "${v_part%% as *}")" ;;
        *)        v_version="$(trim "$v_part")" ;;
    esac
    [ -z "$v_version" ] && v_version="default"
    [ "$v_version" = "master" ] && v_version="default"
    [ -z "$v_label" ] && v_label="$(default_label "$v_stage" "$v_version")"

    [ -f "$SRC_DIR/conv_$v_stage.cpp" ] || die "no such stage file: src/conv_$v_stage.cpp (in 'run $spec')"
    if [ "$v_version" != "default" ]; then
        grep -qE "^[[:space:]]*void[[:space:]]+$v_version[[:space:]]*\(" "$SRC_DIR/conv_$v_stage.cpp" \
            || die "version '$v_version' not found in src/conv_$v_stage.cpp (try --list $v_stage)"
    fi
    v_out="$OUTDIR/${v_stage}_${v_label}_speedup.txt"
    [ -z "${SEEN_OUT[$v_out]:-}" ] \
        || die "two run lines both write $v_out - give one of them a distinct label with 'as <label>'"
    SEEN_OUT[$v_out]=1
done
unset SEEN_OUT; declare -A SEEN_OUT=()

for spec in "${RUN_SPECS[@]}"; do
    # spec: "<stage> : <version> [as <label>] [| sizes=..] [| kernels=..] [| repeat=N] [| seed=N]"
    body="${spec%%|*}"
    overrides="${spec#"$body"}"

    case "$body" in *:*) : ;; *) die "bad run line: 'run $spec' (need '<stage> : <version>')" ;; esac
    stage="$(trim "${body%%:*}")"
    vpart="$(trim "${body#*:}")"

    label=""
    case "$vpart" in
        *" as "*) label="$(trim "${vpart##* as }")"; version="$(trim "${vpart%% as *}")" ;;
        *)        version="$(trim "$vpart")" ;;
    esac
    [ -z "$version" ] && version="default"
    [ "$version" = "master" ] && version="default"
    [ -z "$label" ] && label="$(default_label "$stage" "$version")"

    [ -f "$SRC_DIR/conv_$stage.cpp" ] || die "no such stage file: src/conv_$stage.cpp"

    r_sizes="$G_SIZES"; r_kernels="$G_KERNELS"; r_repeat="$G_REPEAT"; r_seed="$G_SEED"
    if [ -n "$overrides" ]; then
        while IFS= read -r ov; do
            [ -z "$ov" ] && continue
            okey="$(trim "${ov%%=*}")"; oval="$(trim "${ov#*=}")"
            case "$okey" in
                sizes)   r_sizes="$oval" ;;
                kernels) r_kernels="$oval" ;;
                repeat)  r_repeat="$oval" ;;
                seed)    r_seed="$oval" ;;
                *) die "unknown per-run override '$okey' in: run $spec" ;;
            esac
        done < <(printf '%s' "${overrides#|}" | tr '|' '\n')
    fi

    outfile="$OUTDIR/${stage}_${label}_speedup.txt"
    if [ -n "${SEEN_OUT[$outfile]:-}" ]; then
        die "two run lines both write $outfile - give one of them a distinct label with 'as <label>'"
    fi
    SEEN_OUT[$outfile]=1

    info ""
    info "=============================================================="
    info " stage=$stage  version=$version  label=$label"
    info " sizes=[$r_sizes]  kernels=[$r_kernels]  repeat=$r_repeat  seed=$r_seed"
    info " -> $outfile"
    info "=============================================================="

    if [ "$DRY_RUN" -eq 1 ]; then
        while IFS= read -r size; do
            while IFS= read -r kk; do
                info "  would run: bin/conv $stage ${size%%x*} ${size##*x} $kk $r_seed   x$r_repeat"
            done < <(split_list "$r_kernels")
        done < <(split_list "$r_sizes")
        continue
    fi

    info "building ($MAKE clean && $MAKE all) ..."
    bdir="$(stage_and_build "$stage" "$version" "$label")" || exit 1
    bin="$bdir/bin/conv"
    [ -x "$bin" ] || die "expected binary not produced: $bin"

    {
        echo "########################################################################"
        echo "# CS683 PA-1 Task 1  speedup analysis"
        echo "# stage        : $stage        (src/conv_$stage.cpp)"
        echo "# version      : $version"
        echo "# label        : $label"
        echo "# sizes        : $r_sizes"
        echo "# kernel sizes : $r_kernels"
        echo "# seed         : $r_seed"
        echo "# repeats      : $r_repeat"
        echo "# pinned to cpu: ${G_PIN_CPU:-<none>}"
        echo "# host / date  : $HOST / $STARTED"
        echo "# build dir    : $bdir  (Makefile copied verbatim, dispatcher patched)"
        echo "########################################################################"
    } > "$outfile"

    while IFS= read -r size; do
        H="${size%%x*}"; W="${size##*x}"
        case "$H$W" in *[!0-9]*) warn "skipping malformed size '$size'"; continue ;; esac
        if [ $((W % 8)) -ne 0 ]; then
            warn "skipping ${H}x${W}: harness requires W to be a multiple of 8"
            continue
        fi
        while IFS= read -r kk; do
            case "$kk" in *[!0-9]*) warn "skipping malformed kernel size '$kk'"; continue ;; esac
            if [ $((kk % 2)) -ne 1 ]; then
                warn "skipping K=$kk: harness requires K to be odd"
                continue
            fi
            for rep in $(seq 1 "$r_repeat"); do
                info "  run  H=$H W=$W K=$kk  rep $rep/$r_repeat"
                {
                    echo ""
                    echo "----- [$stage/$label] H=$H W=$W K=$kk seed=$r_seed rep=$rep/$r_repeat -----"
                } >> "$outfile"

                out="$("${PIN_PREFIX[@]}" "$bin" "$stage" "$H" "$W" "$kk" "$r_seed" 2>&1)"
                rc=$?
                printf '%s\n' "$out" >> "$outfile"
                if [ $rc -ne 0 ]; then
                    warn "  bin/conv exited with status $rc (output kept in $outfile)"
                    continue
                fi

                # pull the two table rows out for the machine-readable summary
                printf '%s\n' "$out" | awk -v stage="$stage" -v label="$label" \
                    -v version="$version" -v H="$H" -v W="$W" -v K="$kk" \
                    -v seed="$r_seed" -v rep="$rep" -v csv="$CSV" '
                    /^naive \(ref\)/ { naive_ms = $4 }
                    $1 == stage && NF >= 5 {
                        correct = $2; ms = $3; gf = $4; sp = $5; sub(/x$/, "", sp)
                        printf("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                               stage, label, version, H, W, K, seed, rep,
                               correct, naive_ms, ms, gf, sp) >> csv
                        if (correct != "yes")
                            printf("warning: %s/%s H=%s W=%s K=%s reported INCORRECT output\n",
                                   stage, label, H, W, K) > "/dev/stderr"
                        if (ms + 0 == 0)
                            printf("warning: %s/%s H=%s W=%s K=%s timed at 0.000 ms - below the clock resolution, use a larger H/W\n",
                                   stage, label, H, W, K) > "/dev/stderr"
                    }'
            done
        done < <(split_list "$r_kernels")
    done < <(split_list "$r_sizes")

    # per-file footer: median speedup per (size, K) over the repeats
    {
        echo ""
        echo "========================= SUMMARY ($stage / $label) ========================="
        awk -F, -v stage="$stage" -v label="$label" '
            NR == 1 { next }
            $1 == stage && $2 == label {
                key = $4 "x" $5 " K=" $6
                if (!(key in seen)) { order[++n] = key; seen[key] = 1 }
                cnt[key]++; sp[key, cnt[key]] = $13 + 0
                nv[key] += $10; st[key] += $11
                if ($9 != "yes") bad[key] = 1
            }
            END {
                printf("%-18s %8s %12s %12s %10s\n", "workload", "runs", "naive(ms)", "stage(ms)", "speedup")
                printf("---------------------------------------------------------------------\n")
                for (i = 1; i <= n; i++) {
                    k = order[i]; c = cnt[k]
                    for (a = 1; a <= c; a++) for (b = a + 1; b <= c; b++)
                        if (sp[k, b] < sp[k, a]) { t = sp[k, a]; sp[k, a] = sp[k, b]; sp[k, b] = t }
                    med = (c % 2) ? sp[k, (c + 1) / 2] : (sp[k, c / 2] + sp[k, c / 2 + 1]) / 2
                    printf("%-18s %8d %12.3f %12.3f %9.2fx%s\n",
                           k, c, nv[k] / c, st[k] / c, med, (bad[k] ? "  [INCORRECT]" : ""))
                }
                printf("\n(naive/stage columns are means over the runs; speedup is the median.)\n")
            }' "$CSV"
    } >> "$outfile"

    info "wrote $outfile"
done

# --------------------------------------------------------- global summary ----
if [ "$DRY_RUN" -eq 0 ]; then
    SUMMARY="$OUTDIR/summary.txt"
    {
        echo "CS683 PA-1 Task 1  speedup summary"
        echo "config : $CONFIG"
        echo "host   : $HOST"
        echo "started: $STARTED     finished: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        awk -F, '
            NR == 1 { next }
            {
                run = $1 "/" $2; key = $4 "x" $5 " K=" $6
                if (!(run in rseen))  { rorder[++rn] = run; rseen[run] = 1 }
                if (!(key in kseen))  { korder[++kn] = key; kseen[key] = 1 }
                c = ++cnt[run, key]; sp[run, key, c] = $13 + 0
                if ($9 != "yes") bad[run, key] = 1
            }
            END {
                printf("%-28s", "run \\ workload")
                for (j = 1; j <= kn; j++) printf("%16s", korder[j])
                printf("\n")
                printf("----------------------------")
                for (j = 1; j <= kn; j++) printf("----------------")
                printf("\n")
                for (i = 1; i <= rn; i++) {
                    printf("%-28s", rorder[i])
                    for (j = 1; j <= kn; j++) {
                        r = rorder[i]; k = korder[j]; c = cnt[r, k]
                        if (!c) { printf("%16s", "-"); continue }
                        for (a = 1; a <= c; a++) for (b = a + 1; b <= c; b++)
                            if (sp[r, k, b] < sp[r, k, a]) { t = sp[r, k, a]; sp[r, k, a] = sp[r, k, b]; sp[r, k, b] = t }
                        med = (c % 2) ? sp[r, k, (c + 1) / 2] : (sp[r, k, c / 2] + sp[r, k, c / 2 + 1]) / 2
                        printf("%15s%s", sprintf("%.2fx", med), (bad[r, k] ? "!" : " "))
                    }
                    printf("\n")
                }
                printf("\nmedian speedup vs naive over the repeats;  ! = a run reported incorrect output\n")
            }' "$CSV"
    } > "$SUMMARY"

    info ""
    info "=============================================================="
    cat "$SUMMARY"
    info "=============================================================="
    info "per-run logs : $OUTDIR/<stage>_<label>_speedup.txt"
    info "raw records  : $CSV"
    info "summary      : $SUMMARY"

    if [ "$KEEP_BUILD" -eq 0 ]; then
        rm -rf "$BUILD_ROOT"
    else
        info "build trees  : $BUILD_ROOT  (--clean-build to remove)"
    fi
fi
