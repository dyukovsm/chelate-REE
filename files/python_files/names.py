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
TEMPERATURE = 298.0
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
SMALL_EQ_STEPS      =int(500000)# int(1000000)
# MID_EQ_STEPS renamed to follow SIM core naming standard
MID_EQ_STEPS        = int(1000000)
LONG_EQ_STEPS       = int(10000000)
SLOW_OUTPUT         = int(10000) # 10000
NORMAL_CALC         = int(500)

PRO_STEPS             = int(500000)# int(500000)
PRO_FREE_ENERGY_STEPS = int(500000)# int(500000)
FAST_OUTPUT           = int(1000) # 2000
PRO_DHDL              = int(1000)
FAST_CALC             = int(100)

### GROMACS MD steps and output controls
##SMALL_EQ_STEPS      = int(1000)
### MID_EQ_STEPS renamed to follow SIM core naming standard
##MID_EQ_STEPS        = int(1000)
##LONG_EQ_STEPS       = int(1000)
##SLOW_OUTPUT         = int(100)
##NORMAL_CALC         = int(10)
##
##PRO_STEPS             = int(1000)
##PRO_FREE_ENERGY_STEPS = int(1000)
##FAST_OUTPUT           = int(100)
##FAST_CALC             = int(10)

# Cut-off radius in nm
RCUT = 1.4

# Data filenames and locations (moved to names.py to prevent circular imports)
GENERAL_LOCAL_DATA = 'raw_general_data_for'
GENERAL_GLOBAL_DATA = 'aggregate_general_Data'

# Polypeptide PDB file naming
CLEANED_PDB_SUFFIX = "_cleanedPDB"

# Electrostatic and LJ lambda mapping lookup dict (Renamed from ljLam_eleLam_to_initLam)
# First element of the tuple is electrostatic lambda, second is Lennard-Jones lambda
eleLam_ljLam_to_initLam = {
 (0.000, 0.0): 0,
 (0.025, 0.0): 1,
 (0.050, 0.0): 2,
 (0.075, 0.0): 3,
 (0.100, 0.0): 4,
 (0.125, 0.0): 5,
 (0.150, 0.0): 6,
 (0.175, 0.0): 7,
 (0.200, 0.0): 8,
 (0.225, 0.0): 9,
 (0.250, 0.0): 10,
 (0.300, 0.0): 11,
 (0.350, 0.0): 12,
 (0.400, 0.0): 13,
 (0.450, 0.0): 14,
 (0.500, 0.0): 15,
 (0.550, 0.0): 16,
 (0.600, 0.0): 17,
 (0.650, 0.0): 18,
 (0.700, 0.0): 19,
 (0.750, 0.0): 20,
 (0.800, 0.0): 21,
 (0.850, 0.0): 22,
 (0.900, 0.0): 23,
 (0.950, 0.0): 24,
 (0.975, 0.0): 25,
 (1.000, 0.0): 26,
 (1.0,    0.25): 27,    
 (1.0,    0.5):  28,   
 (1.0,    0.75): 29, 
 (1.0,    1.0):  30
    }

# Analysis constants
ANALYSIS_DIR_PREFIX = "000000_analysis"
GENERAL_FILE_PREFIX = "alchemlyb_HFE"
