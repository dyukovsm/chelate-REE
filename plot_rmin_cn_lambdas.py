#!/usr/bin/env python3

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

##############################################################################
# Appearance
##############################################################################

plt.rcParams.update({
    "font.size": 17,
    "axes.linewidth": 1.5,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 15,
    "legend.title_fontsize": 16,
})

##############################################################################
# Only plot these lambda states
##############################################################################

TARGET_STATES = [
    (0.00, 0.00, 1.0),
    (0.25, 0.00, 1.0),
    (0.50, 0.00, 1.0),
    (0.75, 0.00, 1.0),
    (0.90, 0.00, 1.0),
]

##############################################################################
# Rainbow colour map
##############################################################################

colors = [
    "red",
    "orange",
    "gold",
    "limegreen",
    "purple",
]

cmap = LinearSegmentedColormap.from_list(
    "lambda",
    colors,
    N=len(TARGET_STATES),
)

state_colors = {
    state: cmap(i / (len(TARGET_STATES) - 1))
    for i, state in enumerate(TARGET_STATES)
}

##############################################################################
# Helper functions
##############################################################################

def parse_lambda(filename):
    """
    Extract (ELE,LJ,BOND) from filename.
    """
    m = re.search(
        r"lambda_([0-9.]+)_([0-9.]+)_([0-9.]+)_rdf_summary",
        filename,
    )

    if m is None:
        return None

    return (
        float(m.group(1)),
        float(m.group(2)),
        float(m.group(3)),
    )

def state_label(state):
    ele, lj, bond = state
    return f"({ele:g}, {lj:g}, {bond:g})"

##############################################################################
# Read all available files
##############################################################################

all_files = glob.glob("lambda_*_rdf_summary.txt")

data = {}

for filename in all_files:
    state = parse_lambda(filename)

    if state is None:
        continue

    # This correctly ensures ONLY target states are processed
    if state not in TARGET_STATES:
        continue

    print(f"Reading {filename}")

    df = pd.read_csv(
        filename,
        sep=r"\s+",
        engine="python",
    )

    df = df.sort_values("CHARGE")
    data[state] = df

if len(data) == 0:
    raise RuntimeError("No matching lambda files were found.")

print("\nLoaded lambda states:")
for state in sorted(data):
    print(state)

##############################################################################
# Plot one coordination shell
##############################################################################

def plot_shell(ax, shell):
    """
    Modified to accept the 'ax' object directly.
    """
    if shell not in (1, 2):
        raise ValueError("shell must be 1 or 2")

    min_col = f"min_{shell}"
    cn_col = f"cn_{shell}"

    ##########################################################################
    # Plot each lambda state
    ##########################################################################

    for state in TARGET_STATES:
        if state not in data:
            print(f"Skipping missing lambda state {state}")
            continue

        df = data[state]
        color = state_colors[state]
        charge = df["CHARGE"].astype(float)
        cn = df[cn_col].astype(float)
        rmin = df[min_col].astype(float) * 10.0   # nm -> Å

        ##############################################################
        # CN
        ##############################################################
        ax.plot(
            charge,
            cn,
            color=color,
            linewidth=2.8,
            marker="o",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=state_label(state),
            zorder=3,
        )

        ##############################################################
        # Rmin
        ##############################################################
        ax.plot(
            charge,
            rmin,
            color=color,
            linewidth=2.4,
            linestyle="--",
            marker="D",
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.6,
            zorder=2,
        )

    ##########################################################################
    # Axis formatting
    ##########################################################################
    ax.set_xlabel("Charge Multiplier")
    ax.set_ylabel(f"CN$_{{{shell}}}$ / $R_{{min,{shell}}}$ ($\\AA$)")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    ##########################################################################
    # Legend
    ##########################################################################
    ax.legend(
        title="Lambda State (ELE, LJ, BOND)",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=2,
        frameon=True,
    )

##############################################################################
# Marker legend
##############################################################################

from matplotlib.lines import Line2D

def add_marker_legend(ax):
    marker_handles = [
        Line2D(
            [], [], color="black", marker="o", linestyle="-",
            linewidth=2.5, markersize=8, label="CN",
        ),
        Line2D(
            [], [], color="black", marker="D", linestyle="--",
            linewidth=2.5, markersize=7, label=r"$R_{min}$",
        ),
    ]

    legend2 = ax.legend(
        handles=marker_handles,
        title="Quantity",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )
    ax.add_artist(legend2)

##############################################################################
# Generate and Save plots
##############################################################################

# Shell 1
fig1, ax1 = plt.subplots(figsize=(10, 7))
plot_shell(ax1, shell=1)
add_marker_legend(ax1)
plt.tight_layout()
fig1.savefig("CN_Rmin_shell1.png", dpi=1000, bbox_inches="tight")
fig1.savefig("CN_Rmin_shell1.pdf", bbox_inches="tight")
print("Saved CN_Rmin_shell1.png and .pdf")

# Shell 2
fig2, ax2 = plt.subplots(figsize=(10, 7))
plot_shell(ax2, shell=2)
add_marker_legend(ax2)
plt.tight_layout()
fig2.savefig("CN_Rmin_shell2.png", dpi=1000, bbox_inches="tight")
fig2.savefig("CN_Rmin_shell2.pdf", bbox_inches="tight")
print("Saved CN_Rmin_shell2.png and .pdf")

plt.show()