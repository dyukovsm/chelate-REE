#!/usr/bin/env python3
"""
Plot coordination number vs. distance for each metal ion in the 000000_analysis folder.
Uses the legend.txt file to map job IDs to state point parameters.
Now uses relative paths and has a fixed y‑axis limit.
"""

import os
import re
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ----------------------------------------------------------------------
# Use relative paths – script is assumed to be in the same directory
# as legend.txt and the 000000_analysis folder.
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(script_dir, "000000_analysis")
LEGEND_FILE = os.path.join(script_dir, "legend.txt")

# ----------------------------------------------------------------------
# Custom colormaps
cmap_rainbow = plt.cm.rainbow

colors_red_purple = [
    (0.0, 'red'),
    (0.2, 'orange'),
    (0.4, 'yellow'),
    (0.6, 'green'),
    (0.8, 'blue'),
    (1.0, 'purple')
]
cmap_red_purple = LinearSegmentedColormap.from_list('red_purple', colors_red_purple)

cmap_cyan_purple_black = LinearSegmentedColormap.from_list(
    'cyan_purple_black',
    [(0.0, 'cyan'), (0.9, 'purple'), (1.0, 'black')]
)

# ----------------------------------------------------------------------
def parse_legend(legend_path):
    """Parse legend.txt and return dict: job_id -> metadata dict."""
    entries = {}
    job_id_pattern = re.compile(r'^[0-9a-f]{32}')
    with open(legend_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('job'):
                continue
            match = job_id_pattern.match(line)
            if not match:
                continue
            job_id = match.group()
            rest = line[match.end():].strip()
            try:
                meta = ast.literal_eval(rest)
                entries[job_id] = meta
            except Exception:
                print(f"Warning: could not parse line for {job_id}")
                continue
    return entries

# ----------------------------------------------------------------------
def get_rainbow_color(ele):
    """Map lambda_ELE (0..1) to a rainbow colour (red->violet)."""
    ele = min(1.0, max(0.0, ele))
    # Custom colour assignment – keep as you have it
    if ele == 0.9:
        return (np.float64(1.0), np.float64(0.0), np.float64(0.0), np.float64(1.0))
    elif ele == 0.8:
        return (np.float64(1.0), np.float64(0.4), np.float64(0.0), np.float64(1.0))
    elif ele == 0.65:
        return (np.float64(1.0), np.float64(0.88), np.float64(0.1), np.float64(1.0))
    else:
        return cmap_rainbow(ele)

def get_color_for_lj(lj):
    lj = min(1.0, max(0.0, lj))
    return cmap_cyan_purple_black(lj)

def get_red_purple_color(bonded):
    bonded = min(1.0, max(0.0, bonded))
    return cmap_red_purple(bonded)

def get_dash_offset(value, max_offset=9):
    offset = int(round(value * max_offset))
    offset = min(offset, max_offset)
    return (offset, (3.5, 10))

# ----------------------------------------------------------------------
def plot_metal(metal, entries, base_dir):
    subdir_name = f"{metal}_LBT5-_0_False"
    subdir_path = os.path.join(base_dir, subdir_name)
    if not os.path.isdir(subdir_path):
        print(f"Subdirectory {subdir_path} not found, skipping {metal}")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    mini_data = []

    for job_id, meta in entries.items():
        fname = f"{job_id}_debug_rdf.txt"
        fpath = os.path.join(subdir_path, fname)
        if not os.path.isfile(fpath):
            print(f"File {fpath} not found, skipping job {job_id}")
            continue

        try:
            df = pd.read_csv(fpath, sep=r'\s+')
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

        r = df['r_nm'].values
        cn = df['cn'].values

        bonded = meta.get('lambda_BONDED', 0.0)
        ele = meta.get('lambda_ELE', 0.0)
        lj = meta.get('lambda_LJ', 0.0)

        if bonded == 1.0 and ele < 1.0 and lj == 0.0:
            color = get_rainbow_color(ele)
            dash_style = get_dash_offset(ele)
            ax.plot(r, cn, linestyle=dash_style, color=color,
                    label=f'ele={ele:.3f}', linewidth=2.5)

        elif bonded == 1.0 and ele == 1.0 and lj >= 0.0:
            color = get_color_for_lj(lj)
            ax.plot(r, cn, linestyle='None', marker='o', markerfacecolor='none',
                    markeredgecolor=color, markeredgewidth=1.2,
                    label=f'LJ={lj:.2f}')

        if ele == 0.0 and lj == 0.0:
            mini_data.append((bonded, r, cn))

    # ---- Main plot limits ----
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 10)   # fixed upper limit – adjust if needed

    # ---- Create inset if data exists ----
    if mini_data:
        mini_data.sort(key=lambda x: x[0])
        axins = ax.inset_axes([0.70, 0.10, 0.25, 0.25])

        for bonded, r, cn in mini_data:
            color = get_red_purple_color(bonded)
            dash_style = get_dash_offset(bonded)
            axins.plot(r, cn, linestyle=dash_style, color=color,
                       linewidth=1.8)

        axins.set_xlim(0, 0.75)
        axins.set_ylim(0, 10)
        axins.set_xlabel('r (nm)', fontsize=8)
        axins.set_ylabel('CN', fontsize=8)
        axins.tick_params(labelsize=7)
        # Legend removed
    else:
        print(f"No bonded-only data found for {metal}, skipping inset.")

    # ---- Add colorbars ----
    plt.subplots_adjust(right=0.7)

    norm_ele = Normalize(vmin=0, vmax=1)
    sm_ele = ScalarMappable(norm=norm_ele, cmap=cmap_rainbow)
    sm_ele.set_array([])
    cax1 = fig.add_axes([0.75, 0.2, 0.02, 0.6])
    cbar1 = fig.colorbar(sm_ele, cax=cax1, orientation='vertical')
    cbar1.set_label('λ_ELE (bonded=1, LJ=0)', fontsize=10)

    norm_lj = Normalize(vmin=0, vmax=1)
    sm_lj = ScalarMappable(norm=norm_lj, cmap=cmap_cyan_purple_black)
    sm_lj.set_array([])
    cax2 = fig.add_axes([0.85, 0.2, 0.02, 0.6])
    cbar2 = fig.colorbar(sm_lj, cax=cax2, orientation='vertical')
    cbar2.set_label('λ_LJ (bonded=1, ELE=1)', fontsize=10)

    # ---- Main axes labels and title ----
    ax.set_xlabel('Distance r (nm)', fontsize=12)
    ax.set_ylabel('Coordination Number', fontsize=12)
    ax.set_title(f'{metal} – Coordination Number vs. Distance', fontsize=14)
    ax.grid(alpha=0.0)

    # Save the figure
    out_file = os.path.join(subdir_path, f"{metal}_coord_vs_r.png")
    plt.savefig(out_file, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot for {metal} to {out_file}")

# ----------------------------------------------------------------------
def main():
    if not os.path.isfile(LEGEND_FILE):
        print(f"Legend file not found: {LEGEND_FILE}")
        return
    all_entries = parse_legend(LEGEND_FILE)

    metals = {}
    for job_id, meta in all_entries.items():
        metal = meta.get('metal')
        if metal is None:
            continue
        metals.setdefault(metal, {})[job_id] = meta

    for metal, entries in metals.items():
        plot_metal(metal, entries, BASE_DIR)

if __name__ == "__main__":
    main()