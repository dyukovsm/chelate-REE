import os

# Existing names.py contents
#GMX_PREFIX = '/usr/local/gromacs/bin/gmx' # potoff
GMX_PREFIX = 'gmx' # grid/conda

NAME_EQ_NVT = 'EQ_NVT'

# chunked (legacy) workflow mappings in case we need to revert to it when we need
# to run jobs in small chunks in requeue partitions.

# NAME_ELONGATED = 'ELONGATED_BOX_PLACEHOLDER'

# NAME_EQ_CHUNK_COUNT = int(10);  NAME_PRO_CHUNK_COUNT = int(4)
# NAME_EQ_SURFTEN = 'EQ_SURFTEN'; NAME_PRO_SURFTEN = 'PRO_SURFTEN'

#EQ_SURFTEN_CHUNK_TO_STARTING_GRO_FILE = {
#    0 : f'{NAME_ELONGATED}',
#    1 : f'{NAME_EQ_SURFTEN}_CHUNK_1',
#    2 : f'{NAME_EQ_SURFTEN}_CHUNK_2',
#    3 : f'{NAME_EQ_SURFTEN}_CHUNK_3',
#    4 : f'{NAME_EQ_SURFTEN}_CHUNK_4',
#    5 : f'{NAME_EQ_SURFTEN}_CHUNK_5',
#    6 : f'{NAME_EQ_SURFTEN}_CHUNK_6',
#    7 : f'{NAME_EQ_SURFTEN}_CHUNK_7',
#    8 : f'{NAME_EQ_SURFTEN}_CHUNK_8',
#    9 : f'{NAME_EQ_SURFTEN}_CHUNK_9'
#}
#
#PRO_SURFTEN_CHUNK_TO_STARTING_GRO_FILE = {
#    0 : f'{NAME_EQ_SURFTEN}_CHUNK_9',
#    1 : f'{NAME_PRO_SURFTEN}_CHUNK_1',
#    2 : f'{NAME_PRO_SURFTEN}_CHUNK_2',
#    3 : f'{NAME_PRO_SURFTEN}_CHUNK_3'
#}

# --- Restructured & Cleaned Constants from project.py ---

# GROMACS file and prefix names
NAME_EQ_NPT_BERENDSEN = "EQ_NPT_BERENDSEN"
NAME_EQ_CANON = "EQ_CANON"
NAME_PRO_CANON = "PRO_CANON"
NAME_PRE_EQ_NPT_BERENDSEN = "template_PRE_EQ_NPT_BERENDSEN"

# Thermodynamic conditions
TEMPERATURE = 300.0
# Pressure in bar
PRESSURE = 1.0

# Dynamic path resolution to project root directory
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# File and template lists
INIT_FILE_LIST = ['init.gro', 'init.top']
MDP_FILE_LIST = [f'{NAME_EQ_NVT}.mdp', f'{NAME_EQ_NPT_BERENDSEN}.mdp', f'{NAME_EQ_CANON}.mdp', f'{NAME_PRO_CANON}.mdp']

# Metal cation charge specs
METAL_FORMAL_CHARGES = {
    'Ca': 2,
    'Al': 3, 'Fe': 3, 'Cr': 3, 'In': 3, 'Tl': 3,
    'Y': 3, 'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3,
    'Sm': 3, 'Eu': 3, 'Gd': 3, 'Tb': 3, 'Dy': 3,
    'Er': 3, 'Tm': 3, 'Lu': 3,
    'Hf': 4, 'Zr': 4, 'U': 4, 'Pu': 4, 'Th': 4,
}

# GROMACS MD steps and output controls
SMALL_EQ_STEPS      =int(1000000)# int(1000000)
# MID_EQ_STEPS renamed to follow SIM core naming standard
MID_EQ_STEPS        = int(100000)
LONG_EQ_STEPS       = int(500000) # for restraints used 10000000
SLOW_OUTPUT         = int(10000) # for restraints used 1000
NORMAL_CALC         = int(100)

PRO_STEPS             = int(100000)# int(500000)
PRO_FREE_ENERGY_STEPS = int(2000000)# int(500000)
FAST_OUTPUT           = int(5000) # 2000
PRO_DHDL              = int(1000)
FAST_CALC             = int(100)

# Cut-off radius in nm
RCUT = 1.0

# Data filenames and locations (moved to names.py to prevent circular imports)
GENERAL_LOCAL_DATA = 'raw_general_data_for'
GENERAL_GLOBAL_DATA = 'aggregate_general_Data'

# Polypeptide PDB file naming
CLEANED_PDB_SUFFIX = "_cleanedPDB"

# Electrostatic and LJ lambda mapping lookup dict (Renamed from ljLam_eleLam_to_initLam)
# First element of the tuple is electrostatic lambda, second is Lennard-Jones lambda
eleLam_ljLam_to_initLam = {
    #constraint lambda values.
    (0.000,	0.000,	0.000): 0,
    (0.250,	0.000,	0.000): 1,
    (0.500,	0.000,	0.000): 2,
    (0.750,	0.000,	0.000): 3,
    (1.000,	0.000,	0.000): 4,
    (1.000,	0.025,	0.000): 5,
    (1.000,	0.050,	0.000): 6,
    (1.000,	0.075,	0.000): 7,
    (1.000,	0.100,	0.000): 8,
    (1.000,	0.125,	0.000): 9,
    (1.000,	0.150,	0.000): 10,
    (1.000,	0.175,	0.000): 11,
    (1.000,	0.200,	0.000): 12,
    (1.000,	0.250,	0.000): 13,
    (1.000,	0.300,	0.000): 14,
    (1.000,	0.350,	0.000): 15,
    (1.000,	0.400,	0.000): 16,
    (1.000,	0.450,	0.000): 17,
    (1.000,	0.500,	0.000): 18,
    (1.000,	0.550,	0.000): 19,
    (1.000, 0.600,  0.000): 20,
    (1.000,	0.650,	0.000): 21,
    (1.000, 0.700,  0.000): 22,
    (1.000,	0.750,	0.000): 23,
    (1.000,	0.800,	0.000): 24,
    (1.000, 0.835,  0.000): 25,
    (1.000, 0.870,  0.000): 26,
    (1.000,	0.900,	0.000): 27,
    (1.000,	1.000,	0.000): 28,
    (1.000,	1.000,	0.300): 29,
    (1.000,	1.000,	0.600): 30,
    (1.000,	1.000,	0.750): 31,
    (1.000,	1.000,	0.900): 32,
    (1.000,	1.000,	1.000): 33
    }

# Analysis constants
ANALYSIS_DIR_PREFIX = "000000_analysis"
GENERAL_FILE_PREFIX = "alchemlyb_HFE"
