import matplotlib.pyplot as plt
import os

# Matrix sizes (X-axis)
matrix_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Mean speedup values mapped by Tile Size
speedups = {
    8:    [0.72, 0.72, 1.17, 0.72, 0.72, 0.71, 0.72, 0.72, 0.72],
    16:   [0.90, 0.93, 1.17, 0.95, 0.92, 0.92, 0.93, 0.94, 0.92],
    32:   [0.99, 1.12, 1.37, 1.12, 1.10, 1.05, 1.07, 1.06, 0.98],
    64:   [1.16, 1.20, 1.55, 1.17, 1.18, 1.18, 1.17, 1.17, 1.15],
    128:  [1.26, 1.22, 1.40, 1.24, 1.25, 1.20, 1.21, 1.18, 1.20],
    256:  [1.06, 1.27, 1.59, 1.29, 1.27, 1.24, 1.25, 1.28, 1.21],
    512:  [1.25, 1.26, 1.59, 1.32, 1.30, 1.23, 1.26, 1.23, 1.21],
    1024: [1.05, 1.23, 1.22, 1.32, 1.31, 1.27, 1.26, 1.25, 1.23]
}

# Standard deviation values mapped by Tile Size
std_devs = {
    8:    [0.11, 0.13, 0.14, 0.08, 0.07, 0.06, 0.05, 0.04, 0.05],
    16:   [0.14, 0.19, 0.20, 0.09, 0.04, 0.04, 0.03, 0.04, 0.03],
    32:   [0.15, 0.09, 0.17, 0.05, 0.03, 0.05, 0.03, 0.03, 0.04],
    64:   [0.15, 0.09, 0.12, 0.06, 0.04, 0.02, 0.02, 0.02, 0.03],
    128:  [0.07, 0.09, 0.11, 0.05, 0.03, 0.03, 0.02, 0.03, 0.02],
    256:  [0.22, 0.09, 0.06, 0.09, 0.03, 0.03, 0.02, 0.01, 0.03],
    512:  [0.07, 0.07, 0.07, 0.05, 0.03, 0.02, 0.03, 0.03, 0.03],
    1024: [0.43, 0.11, 0.25, 0.05, 0.03, 0.04, 0.02, 0.03, 0.03]
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

save_path = os.path.join(save_dir, 'speedup_plot_repeat10.png')
plt.savefig(save_path)
print(f"Plot successfully saved to: {save_path}")

# Display the plot
plt.show()