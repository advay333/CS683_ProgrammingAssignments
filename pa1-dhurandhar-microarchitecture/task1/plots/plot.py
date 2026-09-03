import matplotlib.pyplot as plt
import os

# Data extracted from image_e8293b.png
matrix_sizes = [1024, 2048, 4096, 8192, 16384]

# Speedup values mapped by Tile Size
speedups = {
    8: [0.78, 0.70, 0.69, 0.68, 0.71],
    16: [1.08, 1.02, 1.03, 1.01, 1.03],
    32: [1.09, 1.02, 1.10, 1.04, 0.90],
    64: [1.19, 0.90, 0.99, 0.97, 1.05],
    128: [1.17, 1.10, 1.00, 1.03, 1.03],
    256: [1.24, 1.13, 1.05, 1.10, 1.06],
    512: [1.18, 1.16, 1.11, 1.11, 1.08],
    1024: [1.19, 1.15, 1.12, 1.08, 1.13]
}

plt.figure(figsize=(10, 6))

# Plot a line for each tile size
for tile_size, values in speedups.items():
    plt.plot(matrix_sizes, values, marker='o', label=f'{tile_size}')

# Axis labels and title
plt.xlabel('Matrix Size')
plt.ylabel('Speedup')
plt.title('Speedup vs. Matrix Size by Tile Size')

# Set x-axis to log base 2 for visual distribution
plt.xscale('log', base=2)
plt.xticks(matrix_sizes, matrix_sizes)

# Add grid and legend (index)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Tile Size', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Determine script directory and save the figure
try:
    # Gets the directory where this script file is located
    save_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback if run in an interactive environment (like Jupyter Notebook)
    save_dir = os.getcwd()

save_path = os.path.join(save_dir, 'speedup_plot_logs2.png')
plt.savefig(save_path)
print(f"Plot successfully saved to: {save_path}")

# Display the plot
plt.show()