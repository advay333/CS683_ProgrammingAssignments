#!/bin/bash

# Define constants for the output location
OUTPUT_DIR="./logs_final_optimized"
OUTPUT_FILE="optimized_speedup.txt"
FULL_LOG_PATH="$OUTPUT_DIR/$OUTPUT_FILE"

# Hardware events to track
PERF_EVENTS="cpu_core/L1-dcache-load-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches"

# Fixed benchmark parameters
KERNEL_SIZE=3
SEED=1234
REPETITIONS=20
START_SIZE=64
END_SIZE=16384

# Ensure the target directory exists before running commands
mkdir -p "$OUTPUT_DIR"

# Notify the user on the terminal (this does NOT go into the file)
echo "Running optimized speed up and saving output to $FULL_LOG_PATH..."
{
    echo "Cleaning bin and compiling again"
    make clean
    make all
    echo "Compilation done"
}
# Group all commands inside { } to redirect stdout and stderr at once
{
    # Iterate over powers of 2 from 64 to 16384
    for ((size = START_SIZE; size <= END_SIZE; size *= 2)); do
        echo "Starting with $size"
        
        for ((rep = 1; rep <= REPETITIONS; rep++)); do
            sudo perf stat -e "$PERF_EVENTS" taskset -c 1 bin/conv optimized "$size" "$size" "$KERNEL_SIZE" "$SEED"
        done
        
        echo "Done with $size"
    done
} > "$FULL_LOG_PATH" 2>&1

echo "Done!"