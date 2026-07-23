#DeepSeek V1
##!/usr/bin/env python3
#"""
#Plot coordination number vs. distance for each metal ion in the 000000_analysis folder.
#Uses the legend.txt file to map job IDs to state point parameters.
#"""
#
#import os
#import re
#import ast
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap, Normalize
#from matplotlib.cm import ScalarMappable
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#
## ----------------------------------------------------------------------
#import os
#script_dir = os.path.dirname(os.path.abspath(__file__))
#BASE_DIR = os.path.join(script_dir, "000000_analysis")
#LEGEND_FILE = os.path.join(script_dir, "legend.txt")
#
## ----------------------------------------------------------------------
## Custom colormaps
## 1) red -> cyan (for lambda_ELE from 0 to 1)
#cmap_red_cyan = LinearSegmentedColormap.from_list('red_cyan', ['red', 'cyan'])
#
## 2) cyan -> purple -> black (for lambda_LJ from 0 to 1)
##    LJ=0.0 -> cyan, LJ=0.9 -> purple, LJ=1.0 -> black
#cmap_cyan_purple_black = LinearSegmentedColormap.from_list(
#    'cyan_purple_black',
#    [(0.0, 'cyan'), (0.9, 'purple'), (1.0, 'black')]
#)
#
## 3) red -> purple (for bonded-only series in the inset)
#cmap_red_purple = LinearSegmentedColormap.from_list('red_purple', ['red', 'purple'])
#
## ----------------------------------------------------------------------
#def parse_legend(legend_path):
#    """Parse legend.txt and return dict: job_id -> metadata dict."""
#    entries = {}
#    job_id_pattern = re.compile(r'^[0-9a-f]{32}')
#    with open(legend_path, 'r') as f:
#        for line in f:
#            line = line.strip()
#            if not line or line.startswith('job'):
#                continue
#            match = job_id_pattern.match(line)
#            if not match:
#                continue
#            job_id = match.group()
#            # the rest should be a dict string
#            rest = line[match.end():].strip()
#            try:
#                meta = ast.literal_eval(rest)
#                entries[job_id] = meta
#            except Exception:
#                print(f"Warning: could not parse line for {job_id}")
#                continue
#    return entries
#
## ----------------------------------------------------------------------
#def get_color_for_ele(ele):
#    """Map lambda_ELE (0..1) to red->cyan."""
#    # Normalize ele to [0,1]
#    return cmap_red_cyan(ele)
#
#def get_color_for_lj(lj):
#    """Map lambda_LJ (0..1) to cyan->purple->black."""
#    # Clamp to [0,1]
#    lj = min(1.0, max(0.0, lj))
#    return cmap_cyan_purple_black(lj)
#
#def get_color_for_bonded(bonded):
#    """Map lambda_BONDED (0..1) to red->purple."""
#    bonded = min(1.0, max(0.0, bonded))
#    return cmap_red_purple(bonded)
#
## ----------------------------------------------------------------------
#def plot_metal(metal, entries, base_dir):
#    """Create a single plot for a given metal."""
#    # Construct subdirectory name (assumes replicate=0, polypeptide='LBT5-', unNested_usesTemplates=False)
#    subdir_name = f"{metal}_LBT5-_0_False"
#    subdir_path = os.path.join(base_dir, subdir_name)
#    if not os.path.isdir(subdir_path):
#        print(f"Subdirectory {subdir_path} not found, skipping {metal}")
#        return
#
#    fig, ax = plt.subplots(figsize=(9, 6))
#
#    # Data containers for the inset (bonded-only series: ELE=0, LJ=0)
#    mini_data = []  # each element: (bonded, r_array, cn_array)
#
#    # Loop over all entries for this metal
#    for job_id, meta in entries.items():
#        # File name pattern: {job_id}_debug_rdf.txt
#        fname = f"{job_id}_debug_rdf.txt"
#        fpath = os.path.join(subdir_path, fname)
#        if not os.path.isfile(fpath):
#            # Try with a wildcard? but we have exact job_id
#            print(f"File {fpath} not found, skipping job {job_id}")
#            continue
#
#        # Read the data
#        try:
#            df = pd.read_csv(fpath, sep=r'\s+')
#        except Exception as e:
#            print(f"Error reading {fpath}: {e}")
#            continue
#
#        # Extract r and cn
#        r = df['r_nm'].values
#        cn = df['cn'].values
#
#        bonded = meta.get('lambda_BONDED', 0.0)
#        ele = meta.get('lambda_ELE', 0.0)
#        lj = meta.get('lambda_LJ', 0.0)
#
#        # ---- Main plot: first set (dashed, bonded=1, ele<1, lj=0) ----
#        if bonded == 1.0 and ele < 1.0 and lj == 0.0:
#            color = get_color_for_ele(ele)
#            ax.plot(r, cn, linestyle='--', color=color,
#                    label=f'ele={ele:.3f}', linewidth=1.5)
#
#        # ---- Main plot: second set (hollow circles, bonded=1, ele=1, lj>=0) ----
#        elif bonded == 1.0 and ele == 1.0 and lj >= 0.0:
#            color = get_color_for_lj(lj)
#            ax.plot(r, cn, linestyle='None', marker='o', markerfacecolor='none',
#                    markeredgecolor=color, markeredgewidth=1.2,
#                    label=f'LJ={lj:.2f}')
#
#        # ---- Inset data: bonded-only series (ele=0, lj=0) ----
#        if ele == 0.0 and lj == 0.0:
#            mini_data.append((bonded, r, cn))
#
#    # If we have mini_data, create the inset
#    if mini_data:
#        # Sort by bonded value for consistent ordering
#        mini_data.sort(key=lambda x: x[0])
#
#        # Inset axes: lower right corner, 30% width/height of main axes
#        axins = inset_axes(ax, width="30%", height="30%", loc='lower right',
#                           bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes)
#
#        for bonded, r, cn in mini_data:
#            color = get_color_for_bonded(bonded)
#            axins.plot(r, cn, color=color, linewidth=1.8, label=f'B={bonded:.2f}')
#
#        axins.set_xlabel('r (nm)', fontsize=8)
#        axins.set_ylabel('CN', fontsize=8)
#        axins.tick_params(labelsize=7)
#        # Optional: add a small legend for the inset (but it might be too crowded)
#        # axins.legend(loc='upper right', fontsize=6)
#    else:
#        print(f"No bonded-only data found for {metal}, skipping inset.")
#
#    # ---- Add colorbars for the two gradients ----
#    # We'll place them to the right of the main plot.
#    # Create two colorbar axes using fig.add_axes ([left, bottom, width, height])
#    # Adjust main plot to make room
#    plt.subplots_adjust(right=0.7)  # leave 30% for colorbars
#
#    # First colorbar: red->cyan for lambda_ELE
#    norm_ele = Normalize(vmin=0, vmax=1)
#    sm_ele = ScalarMappable(norm=norm_ele, cmap=cmap_red_cyan)
#    sm_ele.set_array([])
#    cax1 = fig.add_axes([0.75, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
#    cbar1 = fig.colorbar(sm_ele, cax=cax1, orientation='vertical')
#    cbar1.set_label('λ_ELE (bonded=1, LJ=0)', fontsize=10)
#
#    # Second colorbar: cyan->purple->black for lambda_LJ
#    norm_lj = Normalize(vmin=0, vmax=1)
#    sm_lj = ScalarMappable(norm=norm_lj, cmap=cmap_cyan_purple_black)
#    sm_lj.set_array([])
#    cax2 = fig.add_axes([0.85, 0.2, 0.02, 0.6])
#    cbar2 = fig.colorbar(sm_lj, cax=cax2, orientation='vertical')
#    cbar2.set_label('λ_LJ (bonded=1, ELE=1)', fontsize=10)
#
#    # ---- Main axes labels and title ----
#    ax.set_xlabel('Distance r (nm)', fontsize=12)
#    ax.set_ylabel('Coordination Number', fontsize=12)
#    ax.set_title(f'{metal} – Coordination Number vs. Distance', fontsize=14)
#
#    # Optionally add a grid for readability
#    ax.grid(alpha=0.3)
#
#    # Save the figure
#    out_file = os.path.join(subdir_path, f"{metal}_coord_vs_r.png")
#    plt.savefig(out_file, dpi=150, bbox_inches='tight')
#    plt.close(fig)
#    print(f"Saved plot for {metal} to {out_file}")
#
## ----------------------------------------------------------------------
#def main():
#    # 1. Parse legend
#    if not os.path.isfile(LEGEND_FILE):
#        print(f"Legend file not found: {LEGEND_FILE}")
#        return
#    all_entries = parse_legend(LEGEND_FILE)
#
#    # 2. Group by metal
#    metals = {}
#    for job_id, meta in all_entries.items():
#        metal = meta.get('metal')
#        if metal is None:
#            continue
#        metals.setdefault(metal, {})[job_id] = meta
#
#    # 3. Plot for each metal
#    for metal, entries in metals.items():
#        plot_metal(metal, entries, BASE_DIR)
#
#if __name__ == "__main__":
#    main()



#DeepSeek V2
#!/usr/bin/env python3
#"""
#Plot coordination number vs. distance for each metal ion in the 000000_analysis folder.
#Uses the legend.txt file to map job IDs to state point parameters.
#"""
#
#import os
#import re
#import ast
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap, Normalize
#from matplotlib.cm import ScalarMappable
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#
## ----------------------------------------------------------------------
## Paths (adjust if necessary)
#BASE_DIR = "/Users/richard/Desktop/Labwork/restrained_polypeptides/chelate-REE/000000_analysis"
#LEGEND_FILE = "/Users/richard/Desktop/Labwork/restrained_polypeptides/chelate-REE/legend.txt"
#
## ----------------------------------------------------------------------
## Custom colormaps
## 1) Rainbow (red->orange->yellow->green->blue->violet) for lambda_ELE
##    Use matplotlib's built-in 'rainbow' which goes from red at 0 to violet at 1.
#cmap_rainbow = plt.cm.rainbow
#
## 2) Rainbow red->purple for bonded-only inset
##    Define a linear segment from red (0) to purple (1) through the rainbow colors.
#colors_red_purple = [
#    (0.0, 'red'),
#    (0.2, 'orange'),
#    (0.4, 'yellow'),
#    (0.6, 'green'),
#    (0.8, 'blue'),
#    (1.0, 'purple')
#]
#cmap_red_purple = LinearSegmentedColormap.from_list('red_purple', colors_red_purple)
#
## 3) Cyan->purple->black for lambda_LJ (unchanged)
#cmap_cyan_purple_black = LinearSegmentedColormap.from_list(
#    'cyan_purple_black',
#    [(0.0, 'cyan'), (0.9, 'purple'), (1.0, 'black')]
#)
#
## ----------------------------------------------------------------------
#def parse_legend(legend_path):
#    """Parse legend.txt and return dict: job_id -> metadata dict."""
#    entries = {}
#    job_id_pattern = re.compile(r'^[0-9a-f]{32}')
#    with open(legend_path, 'r') as f:
#        for line in f:
#            line = line.strip()
#            if not line or line.startswith('job'):
#                continue
#            match = job_id_pattern.match(line)
#            if not match:
#                continue
#            job_id = match.group()
#            rest = line[match.end():].strip()
#            try:
#                meta = ast.literal_eval(rest)
#                entries[job_id] = meta
#            except Exception:
#                print(f"Warning: could not parse line for {job_id}")
#                continue
#    return entries
#
## ----------------------------------------------------------------------
#def get_rainbow_color(ele):
#    """Map lambda_ELE (0..1) to a rainbow colour (red->violet)."""
#    ele = min(1.0, max(0.0, ele))
#    return cmap_rainbow(ele)
#
#def get_color_for_lj(lj):
#    """Map lambda_LJ (0..1) to cyan->purple->black."""
#    lj = min(1.0, max(0.0, lj))
#    return cmap_cyan_purple_black(lj)
#
#def get_red_purple_color(bonded):
#    """Map lambda_BONDED (0..1) to red->purple."""
#    bonded = min(1.0, max(0.0, bonded))
#    return cmap_red_purple(bonded)
#
#def get_dash_offset(value, max_offset=5):
#    """
#    Return a linestyle tuple (offset, (1, 10)) where offset is an integer
#    between 0 and max_offset, linearly scaled from value in [0,1].
#    This creates a single dash shifted along the line.
#    """
#    offset = int(round(value * max_offset))
#    offset = min(offset, max_offset)   # clamp
#    return (offset, (1, 10))
#
## ----------------------------------------------------------------------
#def plot_metal(metal, entries, base_dir):
#    """Create a single plot for a given metal."""
#    subdir_name = f"{metal}_LBT5-_0_False"
#    subdir_path = os.path.join(base_dir, subdir_name)
#    if not os.path.isdir(subdir_path):
#        print(f"Subdirectory {subdir_path} not found, skipping {metal}")
#        return
#
#    fig, ax = plt.subplots(figsize=(9, 6))
#
#    # Data containers for the inset (bonded-only series: ELE=0, LJ=0)
#    mini_data = []  # each element: (bonded, r_array, cn_array)
#
#    # Loop over all entries for this metal
#    for job_id, meta in entries.items():
#        fname = f"{job_id}_debug_rdf.txt"
#        fpath = os.path.join(subdir_path, fname)
#        if not os.path.isfile(fpath):
#            # Try with a wildcard? but we have exact job_id
#            print(f"File {fpath} not found, skipping job {job_id}")
#            continue
#
#        try:
#            df = pd.read_csv(fpath, sep=r'\s+')
#        except Exception as e:
#            print(f"Error reading {fpath}: {e}")
#            continue
#
#        r = df['r_nm'].values
#        cn = df['cn'].values
#
#        bonded = meta.get('lambda_BONDED', 0.0)
#        ele = meta.get('lambda_ELE', 0.0)
#        lj = meta.get('lambda_LJ', 0.0)
#
#        # ---- Main plot: first set (dashed, bonded=1, ele<1, lj=0) ----
#        if bonded == 1.0 and ele < 1.0 and lj == 0.0:
#            color = get_rainbow_color(ele)
#            dash_style = get_dash_offset(ele)
#            ax.plot(r, cn, linestyle=dash_style, color=color,
#                    label=f'ele={ele:.3f}', linewidth=1.5)
#
#        # ---- Main plot: second set (hollow circles, bonded=1, ele=1, lj>=0) ----
#        elif bonded == 1.0 and ele == 1.0 and lj >= 0.0:
#            color = get_color_for_lj(lj)
#            ax.plot(r, cn, linestyle='None', marker='o', markerfacecolor='none',
#                    markeredgecolor=color, markeredgewidth=1.2,
#                    label=f'LJ={lj:.2f}')
#
#        # ---- Inset data: bonded-only series (ele=0, lj=0) ----
#        if ele == 0.0 and lj == 0.0:
#            mini_data.append((bonded, r, cn))
#
#    # ---- Main plot limits ----
#    ax.set_xlim(0, 1.2)          # start at 0, end at 1.2 nm
#    ax.set_ylim(bottom=0)        # y starts at 0, top automatic
#
#    # ---- Create inset if data exists ----
#    if mini_data:
#        # Sort by bonded value for consistent ordering
#        mini_data.sort(key=lambda x: x[0])
#
#        # Position inset higher to avoid overlapping x-axis labels.
#        # We use a manual placement in axes coordinates:
#        # x0=0.70, y0=0.55, width=0.25, height=0.25
#        axins = ax.inset_axes([0.70, 0.55, 0.25, 0.25])
#
#        for bonded, r, cn in mini_data:
#            color = get_red_purple_color(bonded)
#            dash_style = get_dash_offset(bonded)
#            axins.plot(r, cn, linestyle=dash_style, color=color,
#                       linewidth=1.8, label=f'B={bonded:.2f}')
#
#        axins.set_xlim(0, 0.75)   # mini plot x from 0 to 0.75
#        axins.set_ylim(bottom=0)
#        axins.set_xlabel('r (nm)', fontsize=8)
#        axins.set_ylabel('CN', fontsize=8)
#        axins.tick_params(labelsize=7)
#    else:
#        print(f"No bonded-only data found for {metal}, skipping inset.")
#
#    # ---- Add colorbars for the two gradients ----
#    plt.subplots_adjust(right=0.7)  # leave 30% for colorbars
#
#    # First colorbar: rainbow for lambda_ELE
#    norm_ele = Normalize(vmin=0, vmax=1)
#    sm_ele = ScalarMappable(norm=norm_ele, cmap=cmap_rainbow)
#    sm_ele.set_array([])
#    cax1 = fig.add_axes([0.75, 0.2, 0.02, 0.6])
#    cbar1 = fig.colorbar(sm_ele, cax=cax1, orientation='vertical')
#    cbar1.set_label('λ_ELE (bonded=1, LJ=0)', fontsize=10)
#
#    # Second colorbar: cyan->purple->black for lambda_LJ
#    norm_lj = Normalize(vmin=0, vmax=1)
#    sm_lj = ScalarMappable(norm=norm_lj, cmap=cmap_cyan_purple_black)
#    sm_lj.set_array([])
#    cax2 = fig.add_axes([0.85, 0.2, 0.02, 0.6])
#    cbar2 = fig.colorbar(sm_lj, cax=cax2, orientation='vertical')
#    cbar2.set_label('λ_LJ (bonded=1, ELE=1)', fontsize=10)
#
#    # ---- Main axes labels and title ----
#    ax.set_xlabel('Distance r (nm)', fontsize=12)
#    ax.set_ylabel('Coordination Number', fontsize=12)
#    ax.set_title(f'{metal} – Coordination Number vs. Distance', fontsize=14)
#    ax.grid(alpha=0.3)
#
#    # Save the figure
#    out_file = os.path.join(subdir_path, f"{metal}_coord_vs_r.png")
#    plt.savefig(out_file, dpi=150, bbox_inches='tight')
#    plt.close(fig)
#    print(f"Saved plot for {metal} to {out_file}")
#
## ----------------------------------------------------------------------
#def main():
#    if not os.path.isfile(LEGEND_FILE):
#        print(f"Legend file not found: {LEGEND_FILE}")
#        return
#    all_entries = parse_legend(LEGEND_FILE)
#
#    metals = {}
#    for job_id, meta in all_entries.items():
#        metal = meta.get('metal')
#        if metal is None:
#            continue
#        metals.setdefault(metal, {})[job_id] = meta
#
#    for metal, entries in metals.items():
#        plot_metal(metal, entries, BASE_DIR)
#
#if __name__ == "__main__":
#    main()



    






#DeepSeek V3
    #!/usr/bin/env python3
"""
Plot coordination number vs. distance for each metal ion in the 000000_analysis folder.
Uses the legend.txt file to map job IDs to state point parameters.
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
# Paths (adjust if necessary)
BASE_DIR = "/Users/richard/Desktop/Labwork/restrained_polypeptides/chelate-REE/000000_analysis"
LEGEND_FILE = "/Users/richard/Desktop/Labwork/restrained_polypeptides/chelate-REE/legend.txt"

# ----------------------------------------------------------------------
# Custom colormaps
# 1) Rainbow (red->orange->yellow->green->blue->violet) for lambda_ELE
cmap_rainbow = plt.cm.rainbow

# 2) Rainbow red->purple for bonded-only inset
colors_red_purple = [
    (0.0, 'red'),
    (0.2, 'orange'),
    (0.4, 'yellow'),
    (0.6, 'green'),
    (0.8, 'blue'),
    (1.0, 'purple')
]
cmap_red_purple = LinearSegmentedColormap.from_list('red_purple', colors_red_purple)

# 3) Cyan->purple->black for lambda_LJ (unchanged)
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
    print(f'ele {ele} cmap_rainbow(ele) {cmap_rainbow(ele)}')
    if ele ==0.9:
        return (np.float64(1.0), np.float64(0.0), np.float64(0.0), np.float64(1.0))
    elif ele ==0.8:
        return (np.float64(1.0), np.float64(0.4), np.float64(0.0), np.float64(1.0))
    elif ele ==0.65:
        return (np.float64(1.0), np.float64(0.88), np.float64(0.1), np.float64(1.0))
    else:
        return cmap_rainbow(ele)

def get_color_for_lj(lj):
    """Map lambda_LJ (0..1) to cyan->purple->black."""
    lj = min(1.0, max(0.0, lj))
    return cmap_cyan_purple_black(lj)

def get_red_purple_color(bonded):
    """Map lambda_BONDED (0..1) to red->purple."""
    bonded = min(1.0, max(0.0, bonded))
    return cmap_red_purple(bonded)

def get_dash_offset(value, max_offset=9):
    """
    Return a linestyle tuple (offset, (1.5, 10)) where offset is an integer
    between 0 and max_offset, linearly scaled from value in [0,1].
    This creates a single dash shifted along the line (dash length 1.5 points).
    """
    offset = int(round(value * max_offset))
    offset = min(offset, max_offset)   # clamp
    return (offset, (3.5, 10))

# ----------------------------------------------------------------------
def plot_metal(metal, entries, base_dir):
    """Create a single plot for a given metal."""
    subdir_name = f"{metal}_LBT5-_0_False"
    subdir_path = os.path.join(base_dir, subdir_name)
    if not os.path.isdir(subdir_path):
        print(f"Subdirectory {subdir_path} not found, skipping {metal}")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    # Data containers for the inset (bonded-only series: ELE=0, LJ=0)
    mini_data = []  # each element: (bonded, r_array, cn_array)

    # Loop over all entries for this metal
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

        # ---- Main plot: first set (dashed, bonded=1, ele<1, lj=0) ----
        if bonded == 1.0 and ele < 1.0 and lj == 0.0:
            color = get_rainbow_color(ele)
            dash_style = get_dash_offset(ele)
            ax.plot(r, cn, linestyle=dash_style, color=color,
                    label=f'ele={ele:.3f}', linewidth=2.5)

        # ---- Main plot: second set (hollow circles, bonded=1, ele=1, lj>=0) ----
        elif bonded == 1.0 and ele == 1.0 and lj >= 0.0:
            color = get_color_for_lj(lj)
            ax.plot(r, cn, linestyle='None', marker='o', markerfacecolor='none',
                    markeredgecolor=color, markeredgewidth=1.2,
                    label=f'LJ={lj:.2f}')

        # ---- Inset data: bonded-only series (ele=0, lj=0) ----
        if ele == 0.0 and lj == 0.0:
            mini_data.append((bonded, r, cn))

    # ---- Main plot limits ----
    ax.set_xlim(0, 1.2)          # start at 0, end at 1.2 nm
    ax.set_ylim(bottom=0)        # y starts at 0, top automatic

    # ---- Create inset if data exists ----
    if mini_data:
        # Sort by bonded value for consistent ordering
        mini_data.sort(key=lambda x: x[0])

        # Position inset: bottom at y=0.5 in axes coordinates
        # [x0, y0, width, height] in axes fraction
        axins = ax.inset_axes([0.70, 0.10, 0.25, 0.25])

        # Plot all bonded-only lines with shifted dashes and red→purple colours
        for bonded, r, cn in mini_data:
            color = get_red_purple_color(bonded)
            dash_style = get_dash_offset(bonded)
            axins.plot(r, cn, linestyle=dash_style, color=color,
                       linewidth=1.8, label=f'B={bonded:.2f}')

        # Set mini‑plot limits
        axins.set_xlim(0, 0.75)
        axins.set_ylim(bottom=0)
        axins.set_xlabel('r (nm)', fontsize=8)
        axins.set_ylabel('CN', fontsize=8)
        axins.tick_params(labelsize=7)

        # Add a small legend inside the mini‑plot (upper left)
        # To avoid overcrowding, we show only a subset? But we have at most 5 lines.
        axins.legend(loc='upper left', fontsize=6, framealpha=0.5)
    else:
        print(f"No bonded-only data found for {metal}, skipping inset.")

    # ---- Add colorbars for the two gradients ----
    plt.subplots_adjust(right=0.7)  # leave 30% for colorbars

    # First colorbar: rainbow for lambda_ELE
    norm_ele = Normalize(vmin=0, vmax=1)
    sm_ele = ScalarMappable(norm=norm_ele, cmap=cmap_rainbow)
    sm_ele.set_array([])
    cax1 = fig.add_axes([0.75, 0.2, 0.02, 0.6])
    cbar1 = fig.colorbar(sm_ele, cax=cax1, orientation='vertical')
    cbar1.set_label('λ_ELE (bonded=1, LJ=0)', fontsize=10)

    # Second colorbar: cyan->purple->black for lambda_LJ
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