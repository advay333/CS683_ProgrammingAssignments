import os
import re
import glob
from collections import defaultdict

def parse_speedup_files(directory="."):
    # Nested dictionary: data[simd_size][matrix_size] = {'naive': [], 'simd': []}
    data = defaultdict(lambda: defaultdict(lambda: {'naive': [], 'simd': []}))
    
    # Regex to capture simd size from filename (e.g., simd_512_speedup.txt)
    file_pattern = re.compile(r'simd_(\d+)_speedup\.txt')
    
    # Iterate over all matching text files in the target directory
    for filepath in glob.glob(os.path.join(directory, "simd_*_speedup.txt")):
        filename = os.path.basename(filepath)
        match = file_pattern.match(filename)
        if not match:
            continue
            
        simd_size = int(match.group(1))
        current_matrix_size = None
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Detect matrix size from the Workload banner
                if line.startswith("=== Workload"):
                    # Extract H value (e.g., H=1024)
                    m = re.search(r'H=(\d+)', line)
                    if m:
                        current_matrix_size = int(m.group(1))
                
                # Extract naive execution time
                elif line.startswith("naive (ref)"):
                    parts = line.split()
                    # Example format: naive (ref)     yes           4.609        4.10      1.00x
                    if len(parts) >= 4 and current_matrix_size:
                        time_ms = float(parts[3])
                        data[simd_size][current_matrix_size]['naive'].append(time_ms)
                        
                # Extract simd execution time
                elif line.startswith("simd"):
                    parts = line.split()
                    # Example format: simd            yes           3.922        4.81      1.18x
                    if len(parts) >= 3 and current_matrix_size:
                        time_ms = float(parts[2])
                        data[simd_size][current_matrix_size]['simd'].append(time_ms)
                        
    return data

def calculate_averages_and_speedup(data):
    results = defaultdict(dict)
    
    for simd_size, matrices in sorted(data.items()):
        for matrix_size, times in sorted(matrices.items()):
            naive_times = times['naive']
            simd_times = times['simd']
            
            # Ensure we have data before calculating averages
            if not naive_times or not simd_times:
                continue
                
            avg_naive = sum(naive_times) / len(naive_times)
            avg_simd = sum(simd_times) / len(simd_times)
            
            # Speedup = Average Naive Time / Average simd Time
            avg_speedup = avg_naive / avg_simd
            
            results[simd_size][matrix_size] = {
                'avg_naive_ms': avg_naive,
                'avg_simd_ms': avg_simd,
                'speedup': round(avg_speedup, 2)
            }
            
    return results

def print_speedup_table(results):
    if not results:
        print("No valid data found to generate table.")
        return

    # Extract and sort unique matrix sizes across all files
    matrix_sizes = set()
    for matrices in results.values():
        matrix_sizes.update(matrices.keys())
    matrix_sizes = sorted(list(matrix_sizes))
    
    # Extract and sort unique simd sizes across all files
    simd_sizes = sorted(list(results.keys()))
    
    # Print the formatted table header
    header = f"{'SPEEDUP':<12}" + "".join([f"{ts:>10}" for ts in simd_sizes])
    print(header)
    
    # Print the table rows corresponding to matrix sizes
    for ms in matrix_sizes:
        row_str = f"{ms:<12}"
        for ts in simd_sizes:
            val = results[ts].get(ms, {}).get('speedup', '-')
            row_str += f"{val:>10}"
        print(row_str)

if __name__ == "__main__":
    # 1. Parse all text files in the current directory
    parsed_data = parse_speedup_files()
    
    # 2. Average the 5 runs and calculate final speedups
    averaged_results = calculate_averages_and_speedup(parsed_data)
    
    # 3. Print the final summary table
    print_speedup_table(averaged_results)