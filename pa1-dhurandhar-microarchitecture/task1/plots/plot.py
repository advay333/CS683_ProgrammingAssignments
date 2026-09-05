import os
import matplotlib.pyplot as plt

# Matrix sizes (X-axis)
matrix_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Mean speedup values mapped by Tile Size
speedups = {
    8:    [0.86, 0.86, 1.33, 0.85, 0.83, 0.82, 0.84, 0.85, 0.84],
    16:   [0.98, 0.95, 1.27, 0.91, 0.92, 0.91, 0.93, 0.94, 0.93],
    32:   [0.93, 0.90, 1.21, 0.91, 0.89, 0.91, 0.91, 0.90, 0.93],
    64:   [1.09, 0.94, 1.22, 0.95, 0.98, 0.95, 0.95, 0.97, 0.99],
    128:  [1.00, 1.03, 1.30, 0.96, 0.99, 0.97, 0.96, 0.99, 1.03],
    256:  [1.00, 1.00, 1.38, 0.95, 1.03, 1.00, 1.03, 0.99, 1.00],
    512:  [0.96, 0.97, 1.39, 0.97, 0.99, 1.00, 0.97, 1.02, 1.01],
    1024: [1.00, 0.98, 1.28, 1.00, 1.02, 1.02, 1.02, 1.01, 1.02]
}

# Standard deviation values mapped by Tile Size
std_devs = {
    8:    [0.12, 0.14, 0.10, 0.07, 0.05, 0.06, 0.05, 0.05, 0.06],
    16:   [0.07, 0.10, 0.12, 0.10, 0.06, 0.05, 0.04, 0.04, 0.06],
    32:   [0.17, 0.17, 0.18, 0.11, 0.06, 0.06, 0.05, 0.05, 0.06],
    64:   [0.10, 0.17, 0.11, 0.09, 0.04, 0.05, 0.05, 0.03, 0.04],
    128:  [0.15, 0.19, 0.17, 0.14, 0.08, 0.06, 0.05, 0.04, 0.06],
    256:  [0.11, 0.16, 0.13, 0.10, 0.09, 0.05, 0.05, 0.07, 0.06],
    512:  [0.16, 0.19, 0.10, 0.12, 0.10, 0.04, 0.05, 0.05, 0.07],
    1024: [0.14, 0.15, 0.12, 0.12, 0.08, 0.08, 0.06, 0.04, 0.07]
}

plt.figure(figsize=(11, 6))

# Plot each tile size series with error bars
for tile_size in speedups:
    plt.errorbar(
        matrix_sizes,
        speedups[tile_size],
        yerr=std_devs[tile_size],
        marker='o',
        capsize=3,
        elinewidth=1,
        label=f'{tile_size}'
    )

# Axis labels and title
plt.xlabel('Matrix Size')
plt.ylabel('Speedup')
plt.title('Speedup vs. Matrix Size by Tile Size (with Standard Deviation)')

# Set x-axis to log base 2 for visual distribution
plt.xscale('log', base=2)
plt.xticks(matrix_sizes, [str(m) for m in matrix_sizes], rotation=30)

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Tile Size', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

# Determine script directory and save the figure
try:
    save_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    save_dir = os.getcwd()

save_path = os.path.join(save_dir, 'speedup_plot_repeat_tile_latest.png')
plt.savefig(save_path)
print(f"Plot successfully saved to: {save_path}")

# Display the plot
plt.show()