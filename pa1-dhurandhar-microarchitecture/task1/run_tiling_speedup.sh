#!/bin/bash

# Define constants for the output location
OUTPUT_DIR="./logs_repeat5"
OUTPUT_FILE="tile_1024_speedup.txt"
FULL_LOG_PATH="$OUTPUT_DIR/$OUTPUT_FILE"

# Ensure the target directory exists before running commands
mkdir -p "$OUTPUT_DIR"

# Notify the user on the terminal (this does NOT go into the file)
echo "Running tiling speed up and saving output to $FULL_LOG_PATH..."

# Group all commands inside { } to redirect their output at once
{
echo "Cleaning bin and compiling again"
make clean
make all
echo "Compilation done"
}
{
echo "Starting with 1024"
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 1024 1024 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 1024 1024 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 1024 1024 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 1024 1024 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 1024 1024 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 1024 1024 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 1024 1024 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 1024 1024 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 1024 1024 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 1024 1024 3 1234
echo "Done with 1024"
echo "Starting with 2048"
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 2048 2048 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 2048 2048 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 2048 2048 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 2048 2048 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 2048 2048 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 2048 2048 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 2048 2048 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 2048 2048 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 2048 2048 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 2048 2048 3 1234
echo "Done with 2048"
echo "Starting with 4096"
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 4096 4096 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 4096 4096 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 4096 4096 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 4096 4096 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 4096 4096 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 4096 4096 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 4096 4096 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 4096 4096 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 4096 4096 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 4096 4096 3 1234
echo "Done with 4096"
echo "Starting with 8192"
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 8192 8192 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 8192 8192 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 8192 8192 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 8192 8192 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 8192 8192 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 8192 8192 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 8192 8192 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 8192 8192 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 8192 8192 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 8192 8192 3 1234
echo "Done with 8192"
echo "Starting with 16384"
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 16384 16384 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 16384 16384 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 16384 16384 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 16384 16384 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 16384 16384 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 16384 16384 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 16384 16384 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 16384 16384 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 16384 16384 3 1234
sudo perf stat -e cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/branch-instructions/,cpu_core/branch-misses/,cpu_core/cache-misses/,cpu_core/cpu-cycles/,cpu_core/instructions/,context-switches taskset -c 0 bin/conv tile 16384 16384 3 1234
echo "Done with 16384"
} > "$FULL_LOG_PATH" 2>&1

echo "Done!"