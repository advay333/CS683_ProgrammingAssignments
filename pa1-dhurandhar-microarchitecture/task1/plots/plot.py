import matplotlib.pyplot as plt
import os

# Matrix sizes (X-axis)
matrix_sizes = [1024, 2048, 4096, 8192, 16384]

# Updated speedup values mapped by Tile Size
speedups = {
    8: [0.71, 0.70, 0.70, 0.70, 0.69],
    16: [1.05, 1.01, 1.02, 1.00, 1.00],
    32: [1.08, 1.05, 1.04, 1.02, 1.04],
    64: [1.04, 0.86, 0.88, 0.89, 0.97],
    128: [1.17, 0.99, 1.04, 1.03, 1.04],
    256: [1.19, 1.03, 1.08, 1.07, 1.07],
    512: [1.21, 1.12, 1.12, 1.13, 1.10],
    1024: [1.21, 1.15, 1.15, 1.14, 1.12]
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

# Add grid and legend
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

save_path = os.path.join(save_dir, 'speedup_plot_repeat10.png')
plt.savefig(save_path)
print(f"Plot successfully saved to: {save_path}")

# Display the plot
plt.show()