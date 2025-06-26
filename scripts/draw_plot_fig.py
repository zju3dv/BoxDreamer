"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 16:32:59
Description:

"""
import numpy as np
import matplotlib.pyplot as plt

# Configure for ICCV style (10pt Times)
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times", "Times New Roman", "CMU Serif"]
plt.style.use("seaborn-v0_8-whitegrid")

# Define data
x_values = np.array([2, 4, 6, 8, 10, 15])

datasets = [
    {
        "name": "ADD-0.1d",
        "y_values": np.array([25.7, 37.6, 42.6, 50.4, 52.7, 54.5]),
        "color": "#7FB3B0",
        "marker": "o",
        "linestyle": "--",
    },
    {
        "name": "ADDs-0.1d",
        "y_values": np.array([58.4, 75.5, 79.5, 85.7, 87.9, 88.7]),
        "color": "#B3927F",
        "marker": "s",
        "linestyle": "-",
    },
    {
        "name": "Proj2D@5px",
        "y_values": np.array([30.7, 51.7, 57.9, 68.1, 73.2, 74.7]),
        "color": "#A694C3",
        "marker": "x",
        "linestyle": "-.",
    },
]

# Create figure - adjust size for two-column format
fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)  # Approximately column width

# Plot lines
for dataset in datasets:
    ax.plot(
        x_values,
        dataset["y_values"],
        color=dataset["color"],
        marker=dataset["marker"],
        markersize=4,  # Reduced marker size
        linestyle=dataset["linestyle"],
        linewidth=1,  # Thinner lines
        markeredgewidth=0.8,  # Thinner marker edges
        markerfacecolor=dataset["color"],
        label=r"\textrm{" + dataset["name"] + "}",
    )

# Set chart properties
ax.set_xlim(2, 15)
ax.set_ylim(20, 95)

# Use 10pt font size to match ICCV requirements
ax.set_xlabel(r"\textrm{Number of Reference Views $n$}", fontsize=10)
ax.set_ylabel(r"\textrm{Succ Rate (\%)}", fontsize=10)
ax.set_title(r"\textrm{Sparse View Robustness on LINEMOD}", fontsize=10)

# Set grid lines
ax.grid(True, linestyle="-", alpha=0.2)

# Add legend with appropriate font size
legend = ax.legend(
    loc="lower right",
    frameon=True,
    fontsize=8,  # Reduced font size for legend
    framealpha=0.9,
)

# Set tick label font size
ax.tick_params(axis="both", which="major", labelsize=8)

# Remove top and right borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig("sparseview-plot.pdf", dpi=300, bbox_inches="tight", format="pdf")
plt.savefig("sparseview-plot.png", dpi=300, bbox_inches="tight")

# Show figure
plt.show()
