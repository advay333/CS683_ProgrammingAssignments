#!/bin/bash
# Sweeps the tile size in src/conv_tile.cpp over powers of 2 (8..1024), rebuilding
# and running the full image-size benchmark for each, into its own log file.

set -u

# ---------------------------------------------------------------- configuration
SRC="src/conv_tile.cpp"
OUTPUT_DIR="./logs_final_tile_new"

PERF_EVENTS="cpu_core/L1-dcache-load-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches"

KERNEL_SIZE=3
SEED=1234
REPETITIONS=20
START_SIZE=64
END_SIZE=16384

TILE_START=8
TILE_END=1024

# ------------------------------------------------------------------- safeguards
if [[ ! -f "$SRC" ]]; then
    echo "error: $SRC not found. Run this script from the project root." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Keep a pristine copy of the source and always put it back, even on Ctrl-C or error.
BACKUP="$(mktemp)"
cp "$SRC" "$BACKUP"
restore() {
    cp "$BACKUP" "$SRC"
    rm -f "$BACKUP"
    echo "Restored original $SRC"
}
trap restore EXIT

# Fail early if the line we intend to rewrite is not there in the expected form.
if ! grep -Eq '^[[:space:]]*int[[:space:]]+tile_size[[:space:]]*=[[:space:]]*[0-9]+[[:space:]]*;' "$SRC"; then
    echo "error: could not find a 'int tile_size = <N>;' definition in $SRC" >&2
    exit 1
fi

# Ask for the sudo credential once up front so the loop is not interrupted later.
sudo -v || exit 1

# ------------------------------------------------------------------- the sweep
for ((tile = TILE_START; tile <= TILE_END; tile *= 2)); do

    FULL_LOG_PATH="$OUTPUT_DIR/tile_${tile}_speedup.txt"
    echo "=== tile_size=$tile -> $FULL_LOG_PATH ==="

    # Patch the global tile size in place.
    sed -i -E \
        "s/^[[:space:]]*int[[:space:]]+tile_size[[:space:]]*=[[:space:]]*[0-9]+[[:space:]]*;/int tile_size=${tile};/" \
        "$SRC"

    # Verify the patch actually landed before spending an hour benchmarking it.
    if ! grep -q "^int tile_size=${tile};" "$SRC"; then
        echo "error: failed to set tile_size=$tile in $SRC" >&2
        exit 1
    fi

    {
        echo "############################################################"
        echo "# tile_size = $tile"
        echo "# started   : $(date -Is)"
        echo "############################################################"

        echo "Cleaning bin and compiling again"
        make clean
        make all
        echo "Compilation done"

        for ((size = START_SIZE; size <= END_SIZE; size *= 2)); do
            echo "Starting with $size"

            for ((rep = 1; rep <= REPETITIONS; rep++)); do
                sudo perf stat -e "$PERF_EVENTS" \
                    taskset -c 0 bin/conv tile "$size" "$size" "$KERNEL_SIZE" "$SEED"
            done

            echo "Done with $size"
        done

        echo "# finished  : $(date -Is)"
    } > "$FULL_LOG_PATH" 2>&1

    echo "Done with tile_size=$tile"
done

echo "Sweep complete. Logs in $OUTPUT_DIR/"