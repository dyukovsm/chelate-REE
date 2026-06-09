import os

# Existing names.py contents
GMX_PREFIX = '/usr/local/gromacs/bin/gmx' # potoff
#GMX_PREFIX = 'gmx' # grid

NAME_EQ_NVT = 'EQ_NVT'
NAME_EQ_SURFTEN = 'EQ_SURFTEN'; NAME_PRO_SURFTEN = 'PRO_SURFTEN'

NAME_ELONGATED = 'ELONGATED_BOX_PLACEHOLDER'

NAME_EQ_CHUNK_COUNT = int(10)
NAME_PRO_CHUNK_COUNT = int(4)

# chunked

EQ_SURFTEN_CHUNK_TO_STARTING_GRO_FILE = {
    0 : f'{NAME_ELONGATED}',
    1 : f'{NAME_EQ_SURFTEN}_CHUNK_1',
    2 : f'{NAME_EQ_SURFTEN}_CHUNK_2',
    3 : f'{NAME_EQ_SURFTEN}_CHUNK_3',
    4 : f'{NAME_EQ_SURFTEN}_CHUNK_4',
    5 : f'{NAME_EQ_SURFTEN}_CHUNK_5',
    6 : f'{NAME_EQ_SURFTEN}_CHUNK_6',
    7 : f'{NAME_EQ_SURFTEN}_CHUNK_7',
    8 : f'{NAME_EQ_SURFTEN}_CHUNK_8',
    9 : f'{NAME_EQ_SURFTEN}_CHUNK_9'
}

PRO_SURFTEN_CHUNK_TO_STARTING_GRO_FILE = {
    0 : f'{NAME_EQ_SURFTEN}_CHUNK_9',
    1 : f'{NAME_PRO_SURFTEN}_CHUNK_1',
    2 : f'{NAME_PRO_SURFTEN}_CHUNK_2',
    3 : f'{NAME_PRO_SURFTEN}_CHUNK_3'
}

# --- Restructured & Cleaned Constants from project.py ---

# GROMACS file and prefix names
NAME_EM = "em"
NAME_EQ_NPT_BERENDSEN = "EQ_NPT_BERENDSEN"
NAME_EQ_CANON = "EQ_CANON"
NAME_PRO_CANON = "PRO_CANON"

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
    'Al': 3, 'Fe': 3, 'Cr': 3, 'In': 3, 'Tl': 3,
    'Y': 3, 'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3,
    'Sm': 3, 'Eu': 3, 'Gd': 3, 'Tb': 3, 'Dy': 3,
    'Er': 3, 'Tm': 3, 'Lu': 3,
    'Hf': 4, 'Zr': 4, 'U': 4, 'Pu': 4, 'Th': 4,
}

# GROMACS MD steps and output controls
SMALL_EQ_STEPS      = int(1000000)
MID_EQ_STEPS        = int(2000000)
LONG_EQ_STEPS       = int(4000000)
SLOW_OUTPUT         = int(20000)
NORMAL_CALC         = int(100)

PRO_STEPS             = int(500000)
PRO_FREE_ENERGY_STEPS = int(500000)
FAST_OUTPUT           = int(1000)
FAST_CALC             = int(100)

# Cut-off radius in nm
RCUT = 1.4

# Electrostatic and LJ lambda mapping lookup dict (Renamed from ljLam_eleLam_to_initLam)
# First element of the tuple is electrostatic lambda, second is Lennard-Jones lambda
eleLam_ljLam_to_initLam = {
    (0.0, 0.00): 0,
    (0.2, 0.00): 1,
    (0.4, 0.00): 2,
    (0.6, 0.00): 3,
    (0.8, 0.00): 4,
    (0.9, 0.00): 5,
    (1.0, 0.00): 6,
    (1.0, 0.20): 7,
    (1.0, 0.40): 8,
    (1.0, 0.55): 9,
    (1.0, 0.60): 10,
    (1.0, 0.65): 11,
    (1.0, 0.70): 12,
    (1.0, 0.75): 13,
    (1.0, 0.80): 14,
    (1.0, 0.90): 15,
    (1.0, 1.00): 16
}
