import os
import math
import shutil
import subprocess
from collections import defaultdict
from jinja2 import Environment, FileSystemLoader
import numpy as np
import parmed
from parmed import gromacs
import mbuild as mb
from foyer import Forcefield
import foyer
import pandas as pd
import random
import time
import signac
from flow import FlowProject, aggregator
from flow.environment import DefaultSlurmEnvironment
import flow
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import glob

from files.python_files import names
from files.python_files import job_tester

def simple_mdp_writer(job, mdp_name, parameters, constraints=None, templates_dir=None, template_name=None):
   loader = FileSystemLoader('.')
   env = Environment(loader=loader)
   path = os.path.relpath(f'{templates_dir}')
   MDP_NAME = template_name
   
   if constraints is None:
       update_dict = {
           'constraints_string' : ';',
           'constraints' : 'whatever',
           'constraint_algorithm_string' : ';',
           'constraint_algorithm' : 'whatever',
           'lincs_order_string' : ';',
           'lincs_order' : 'whatever'
       }
   elif 'lincs' in constraints:
       update_dict = {
           'constraints_string' : 'constraints         = ',
           'constraints' : 'all-bonds',
           'constraint_algorithm_string' : 'constraint-algorithm = ',
           'constraint_algorithm' : 'LINCS',
           'lincs_order_string' : 'lincs-order           = ',
           'lincs_order' : '4'
       }
   elif 'shake' in constraints:
       update_dict = {
           'constraints_string' : 'constraints         = ',
           'constraints' : 'all-angles',
           'constraint_algorithm_string' : 'constraint-algorithm = ',
           'constraint_algorithm' : 'SHAKE',
           'lincs_order_string' : 'shake-tol           = ',
           'lincs_order' : '0.00001'
       }
   parameters.update(update_dict)
   
   template_data = parameters
   template = env.get_template(f'{path}/{MDP_NAME}')
   
   output = template.render(template_data)
   with open(f'workspace/{job}/{mdp_name}','w') as f:
       f.write(output)


def gimme_dir(job):
   current_dir = os.getcwd()
   job_dir = f'{current_dir}/workspace/{job}' 
   return current_dir, job


def write_gmxINDEX_forRESIDUES(job, top_file='init.top', gro_file='init.gro', index_file_name='whacky_index_file.ndx'):
   with(job):
       system_pmdTop = gromacs.GromacsTopologyFile(f'{top_file}')
       gmx_gro = gromacs.GromacsGroFile.parse(f'{gro_file}')
       system_pmdTop.box = gmx_gro.box
       system_pmdTop.positions = gmx_gro.positions

       angles4Gromacs = open(f'{index_file_name}','w')
       angles4Gromacs.write('[ WAT ] ;index1, atom_type\n')
       some_angles_written = False
       for i in system_pmdTop.residues:
           comments = [] 
           for j in i.atoms:
               correct_index = j.idx + 1
               comments.append(j)
               angles4Gromacs.write(f'{correct_index}\t')
               some_angles_written = True
           angles4Gromacs.write(f' \t;\t{str(comments)}\n')

       if some_angles_written:
           angles4Gromacs.write('\n;index file written correctly \n')
       angles4Gromacs.close() 


def manual_gmx_index_file_make(job, gro_file='init.gro', index_file_name='whacky_index_file.ndx', skip_residues_from_ncompounds=1000):
   with(job):
       skip_guess = skip_residues_from_ncompounds
       skip_guess = len(str(skip_guess))

       with open(f"{gro_file}", 'r') as f:
           for _ in range(2):
               next(f)  # skip first two lines
           line = f.readline()

       end_positions = [i for i, char in enumerate(line) if char != ' ' and (i == len(line) - 1 or line[i+1] == ' ')]
       column_widths = [end_positions[0] + 1] + [end_positions[i] - end_positions[i-1] for i in range(1, len(end_positions))]

       data = np.genfromtxt(f"{gro_file}", dtype=None, skip_header=2, delimiter=column_widths, encoding='utf-8')
       data = data[:-1]
       print(data)

       result_dict = defaultdict(list)
       for record in data:
           key = record['f0'].strip()
           value = record['f2']
           result_dict[key].append(value)

       index_file = open(f'{index_file_name}','w')
       header_preper = record['f0'].strip()
       header_preper = header_preper[skip_guess:-1]
       index_file.write(f'[ {header_preper} ]\n')

       for i in result_dict.keys():
           dummy_list = result_dict[i]
           for j in dummy_list:
               index_file.write(f'{j}\t')
           index_file.write(f'\t ; {i} \n')

       index_file.close()


def gmx_density_profile(job, trr_or_gro, index_file, tpr_file, output_xvg_name, first_frame, last_frame, slices=128):
   with(job):
       subprocess.run((f'{names.GMX_PREFIX}') + str(' density -f ') + str(f'{trr_or_gro}') + str(' -n ') + str(f'{index_file}') + str(' -s ') + str(f'{tpr_file}') + str(' -o ') + str(f'{output_xvg_name}') + str(' -sl ') + str(f'{slices}'), shell=True)


def give_name_return_whichChunk(job, chunk_dict):
   with(job):
       last_chunk = 0
       for key in chunk_dict.keys():
           print(f'last_chunk : {last_chunk}')
           working_key = key+1
           input_log_file = f'{chunk_dict[working_key]}.log'
           if os.path.isfile(input_log_file):
               if job_tester.look_in_file(job, [input_log_file], "Finished", check_for_not=True, check_for_not_str='Received the TERM'):
                   last_chunk = last_chunk + 1
               else:
                   break
           else:
               break
       return last_chunk


def calculate_free_energy(target_dir):
   import sys
   import os
   import subprocess
   
   current_cwd = os.getcwd()
   os.chdir(target_dir)
   original_stdout = sys.stdout
   original_stderr = sys.stderr
   log_file = open("alchemlyb_log.txt", "w")
   sys.stdout = log_file
   sys.stderr = log_file
   
   # Save original environment variables so we don't break GROMACS subprocesses that need GPU
   orig_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
   orig_jax_x4 = os.environ.get("JAX_ENABLE_X64")
   orig_jax_plat = os.environ.get("JAX_PLATFORMS")
   
   # Test if JAX GPU is functional in a subprocess to absorb any segfaults
   test_code = (
       "import os\n"
       "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n"
       "import jax\n"
       "try:\n"
       "    jax.devices('gpu')\n"
       "except Exception:\n"
       "    sys.exit(1)\n"
   )
   result = subprocess.run([sys.executable, "-c", test_code], capture_output=True)
   need_cpu_fallback = (result.returncode != 0)
   
   if need_cpu_fallback:
       os.environ["CUDA_VISIBLE_DEVICES"] = ""
       os.environ["JAX_ENABLE_X64"] = "True"
       os.environ["JAX_PLATFORMS"] = "cpu"

   try:
       # my goal is to run the worklow manually calling the functions directly.
       #from alchemtest.gmx import load_ABFE
       #from alchemlyb.workflows import ABFE

       # my imports
       import pdb
       from alchemlyb.parsing.gmx import extract_u_nk
       from alchemlyb.parsing.gmx import extract_dHdl
       from alchemlyb.preprocessing.subsampling import decorrelate_u_nk
       from alchemlyb.preprocessing.subsampling import decorrelate_dhdl
       import pandas as pd
       from pathlib import Path
       from alchemlyb.estimators import MBAR, TI
       from alchemlyb.postprocessors.units import get_unit_converter
       from alchemlyb.convergence import forward_backward_convergence
       from alchemlyb.visualisation import plot_convergence, plot_dF_state, plot_mbar_overlap_matrix, plot_ti_dhdl

       #dir = os.path.dirname(load_ABFE()['data']['complex'][0]) # just a dir name
       dir = target_dir

       UNIT_INPUT = 'kJ/mol'
       SIMULATION_TEMPERATURE = names.TEMPERATURE

       file_pattern = '**/PRO_CANON_*.xvg'

       GENERAL_FILE_PREFIX = names.GENERAL_FILE_PREFIX

       TI_SUFFIX = 'TI'; MBAR_SUFFIX = 'MBAR'

       CONVERGNCE_ROOT = 'convergence'; OVERLAP_ROOT = 'overlap'
       STATE_ROOT = 'state'; DH_DL_ROOT = 'dhdl'

       list_of_xvg_files = list(map(str, Path(dir).glob(file_pattern)))

       u_nk_list = []; decorrelated_u_nk_list = [] 
       dHdl_list = []; decorrelated_dhdl_list = [] 

       for file in list_of_xvg_files:
           u_nk_list.append(extract_u_nk(file, T=SIMULATION_TEMPERATURE))
           dHdl_list.append(extract_dHdl(file, T=SIMULATION_TEMPERATURE))

       column_names = u_nk_list[0].columns.values.tolist()

       list_to_ensure_files_allign_with_columnLambas = sorted(
           range(len(list_of_xvg_files)),
           key = lambda x: column_names.index(u_nk_list[x].reset_index("time").index.values[0])
       )

       # apply the sorting so that the files appear in the order of increasing lambda state.
       u_nk_list = [u_nk_list[i] for i in list_to_ensure_files_allign_with_columnLambas]
       dHdl_list = [dHdl_list[i] for i in list_to_ensure_files_allign_with_columnLambas]

       for index_of_unk, u_nk in enumerate(u_nk_list):
           decorrelated_u_nk_list.append(decorrelate_u_nk(u_nk))

       for index_of_unk, dHdl in enumerate(dHdl_list):
           decorrelated_dhdl_list.append(decorrelate_dhdl(dHdl))

       cat_decorrelated_u_nk_list = pd.concat(decorrelated_u_nk_list)
       cat_decorrelated_dhdl_list = pd.concat(decorrelated_dhdl_list)

       mbar_result = MBAR().fit(cat_decorrelated_u_nk_list)
       dHdl_result = TI().fit(cat_decorrelated_dhdl_list)

       ## PERFORM UNFATHOMABLE SORTING OF EACH SIMULATION AND ITS LAMBDA NEIGHBORS
       ## lines 504-611 of abfe.py; from alchemlyb/workflows/abfe.py
       ## black box of not understanding the code starts here.

       num_mbar_states = len(mbar_result.states_)
       num_dHdl_states = len(dHdl_result.states_)

       # Verify coupling state from any .mdp file in the directory
       import glob
       mdp_files = glob.glob('*.mdp')
       couple_lambda0 = None
       couple_lambda1 = None

       if mdp_files:
           mdp_file = mdp_files[0]
           print(f"Reading MDP file: {mdp_file}")
           with open(mdp_file, 'r') as f:
               for line in f:
                   line = line.strip()
                   if line.startswith(';') or line.startswith('#'):
                       continue
                   if 'couple-lambda0' in line:
                       parts = line.split('=')
                       if len(parts) > 1:
                           couple_lambda0 = parts[1].split(';')[0].strip().lower()
                   elif 'couple-lambda1' in line:
                       parts = line.split('=')
                       if len(parts) > 1:
                           couple_lambda1 = parts[1].split(';')[0].strip().lower()
           print(f"Found couple-lambda0: {couple_lambda0}")
           print(f"Found couple-lambda1: {couple_lambda1}")
       else:
           print("Warning: No .mdp files found in the current directory. Cannot automatically verify the coupling state order.")
           print("Defaulting to sign flip (assuming state 0 is coupled and state 18 is decoupled).")

       # Determine sign flip
       sign_flip = True  # Default assuming state 18 is decoupled
       if couple_lambda0 is not None or couple_lambda1 is not None:
           if couple_lambda0 == 'none':
               sign_flip = False
           elif couple_lambda1 == 'none':
               sign_flip = True

       # Conversion factors: kT to kJ/mol
       conversion_factor = SIMULATION_TEMPERATURE * 8.314 / 1000.0
       sign_mult = -1.0 if sign_flip else 1.0
       total_factor = conversion_factor * sign_mult

       print(f"Conversion factor: {conversion_factor:.6f} kJ/mol per kT")
       print(f"Sign flip applied: {sign_flip} (multiplier: {sign_mult})")

       data_dict = {"name": [], "state": []}

       for i in range(num_mbar_states-1):
           data_dict["name"].append(str(i) + " -- " + str(i + 1))
           data_dict["state"].append("States")

       stages =  u_nk_list[0].reset_index("time").index.names

       for stage in stages:
           data_dict["name"].append(stage.split("-")[0])
           data_dict["state"].append("Stages")

       data_dict["name"].append("TOTAL")
       data_dict["state"].append("Stages")

       col_names = ['MBAR','MBAR_Error']
       mbar_delta_f_ = mbar_result.delta_f_
       mbar_d_delta_f_ = mbar_result.d_delta_f_

       data_dict['MBAR'] = []
       data_dict['MBAR_Error'] = []

       for index in range(1, num_mbar_states):
           data_dict['MBAR'].append(mbar_delta_f_.iloc[index - 1, index] * total_factor)
           data_dict['MBAR_Error'].append(mbar_d_delta_f_.iloc[index - 1, index] * conversion_factor)

       for index, stage in enumerate(stages):
           if len(stages) == 1:
               start = 0
               end = num_mbar_states - 1
           else:
               lambda_min = min([state[index] for state in mbar_result.states_])
               lambda_max = max([state[index] for state in mbar_result.states_])
               if lambda_min == lambda_max:
                   start = 0
                   end = 0
               else:
                   states = [state[index] for state in mbar_result.states_]    
                   start = list(reversed(states)).index(lambda_min)
                   start = num_mbar_states - start - 1
                   end = states.index(lambda_max)

           result = mbar_delta_f_.iloc[start, end] * total_factor
           error = mbar_d_delta_f_.iloc[start, end] * conversion_factor
           data_dict['MBAR'].append(result)
           data_dict['MBAR_Error'].append(error)

       # Appending TOTAL (the last row)
       total_result = mbar_delta_f_.iloc[0, num_mbar_states - 1] * total_factor
       total_error = mbar_d_delta_f_.iloc[0, num_mbar_states - 1] * conversion_factor
       data_dict['MBAR'].append(total_result)
       data_dict['MBAR_Error'].append(total_error)

       summary = pd.DataFrame.from_dict(data_dict)
       summary = summary.set_index(["state", "name"])
       summary.index.names = [None, None]
       summary = summary.to_string()

       summary_txt = open(f'{GENERAL_FILE_PREFIX}.txt','w')
       summary_txt.write(f'Free energy results in kJ/mol (converted from kT at T = {SIMULATION_TEMPERATURE} K).\n')
       if not mdp_files:
           summary_txt.write(f'WARNING: No .mdp files found in the current directory. Assuming state 0 is coupled and state 18 is decoupled (sign flip applied).\n')
       summary_txt.write(f'Sign flip applied: {sign_flip} (Target: Solvation/Hydration free energy).\n\n')
       summary_txt.write(f'{summary}\n\n\n______________________________________________________________\n\n\n')

       ######______________________________________________________________######

       data_dict_ti = {"name": [], "state": []}

       for i in range(num_dHdl_states-1):
           data_dict_ti["name"].append(str(i) + " -- " + str(i + 1))
           data_dict_ti["state"].append("States")

       for stage in stages:
           data_dict_ti["name"].append(stage.split("-")[0])
           data_dict_ti["state"].append("Stages")

       data_dict_ti["name"].append("TOTAL")
       data_dict_ti["state"].append("Stages")

       col_names = ['TI','TI_Error']
       dHdl_delta_f_ = dHdl_result.delta_f_
       dHdl_d_delta_f_ = dHdl_result.d_delta_f_

       data_dict_ti['TI'] = []
       data_dict_ti['TI_Error'] = []

       for index in range(1, num_dHdl_states):
           data_dict_ti['TI'].append(dHdl_delta_f_.iloc[index - 1, index] * total_factor)
           data_dict_ti['TI_Error'].append(dHdl_d_delta_f_.iloc[index - 1, index] * conversion_factor)

       for index, stage in enumerate(stages):
           if len(stages) == 1:
               start = 0
               end = num_dHdl_states - 1
           else:
               lambda_min = min([state[index] for state in dHdl_result.states_])
               lambda_max = max([state[index] for state in dHdl_result.states_])
               if lambda_min == lambda_max:
                   start = 0
                   end = 0
               else:
                   states = [state[index] for state in dHdl_result.states_]    
                   start = list(reversed(states)).index(lambda_min)
                   start = num_dHdl_states - start - 1
                   end = states.index(lambda_max)

           result = dHdl_delta_f_.iloc[start, end] * total_factor
           error = dHdl_d_delta_f_.iloc[start, end] * conversion_factor
           data_dict_ti['TI'].append(result)
           data_dict_ti['TI_Error'].append(error)

       # Appending TOTAL (the last row)
       total_result_ti = dHdl_delta_f_.iloc[0, num_dHdl_states - 1] * total_factor
       total_error_ti = dHdl_d_delta_f_.iloc[0, num_dHdl_states - 1] * conversion_factor
       data_dict_ti['TI'].append(total_result_ti)
       data_dict_ti['TI_Error'].append(total_error_ti)

       summary_ti = pd.DataFrame.from_dict(data_dict_ti)
       summary_ti = summary_ti.set_index(["state", "name"])
       summary_ti.index.names = [None, None]
       summary_ti = summary_ti.to_string()

       summary_txt.write(f'\n\n\n')
       summary_txt.write(f'______________________________________________________________')
       summary_txt.write(f'\n\n\n')
       summary_txt.write(f'{summary_ti}')
       summary_txt.close()

       #summary_txt = open(f'{GENERAL_FILE_PREFIX}.txt','w')

       ## print(f"len(data_dict['name']): {len(data_dict['name'])}")
       ## print(f"len(data_dict['state']): {len(data_dict['state'])}")
       ## 
       ## print(f"len(data_dict['TI']): {len(data_dict['TI'])}")
       ## print(f"len(data_dict['TI_Error']): {len(data_dict['TI_Error'])}")

       ## black box of not understanding the code ends here.

       ax = plot_mbar_overlap_matrix(mbar_result.overlap_matrix)
       ax.figure.savefig(f'{GENERAL_FILE_PREFIX}_{OVERLAP_ROOT}_{MBAR_SUFFIX}.png')
       fig = plot_dF_state([mbar_result,dHdl_result],units=UNIT_INPUT)
       fig.savefig(f'{GENERAL_FILE_PREFIX}_{STATE_ROOT}_{TI_SUFFIX}with{MBAR_SUFFIX}.png')

       #del ax, fig

       #convergence = forward_backward_convergence(u_nk_list, 'MBAR', num=min(10, len(u_nk_list[0])))

       #unit_converted_convergence = get_unit_converter(UNIT_INPUT)(convergence)
       #unit_converted_convergence["data_fraction"] = convergence["data_fraction"]

       #ax = plot_convergence(convergence, units=UNIT_INPUT)
       #ax.figure.savefig(f'{GENERAL_FILE_PREFIX}_{CONVERGNCE_ROOT}_{MBAR_SUFFIX}.png')

       #del ax

       ax = plot_ti_dhdl(dHdl_result, units=UNIT_INPUT)
       ax.figure.savefig(f'{GENERAL_FILE_PREFIX}_{DH_DL_ROOT}_{TI_SUFFIX}.png')


       ## states_mbar = mbar_result.states_; mbar_HFE = mbar_result.delta_f_; mbar_fluctuation = mbar_result.d_delta_f_
       ## states_dHdl = dHdl_result.states_; dHdl_HFE = dHdl_result.delta_f_; dHdl_fluctuation = dHdl_result.d_delta_f_
       ## 
       ## 
       ## print(f'using MBAR')
       ## for index, value in enumerate(states_mbar): #range(0, num_states_mbar):
       ##     print(f'init_lambda_state: {index} states_mbar: {states_mbar.iloc[]} HFE: {mbar_HFE} fluctuation: {mbar_fluctuation}')
       ##     
       ##     
       ## print(f'using dHdl (TI)')
       ## for index, value in enumerate(states_dHdl): #range(0, num_states_dHdl):
       ##     print(f'init_lambda_state: {index} HFE: {dHdl_HFE} fluctuation: {dHdl_fluctuation}')

       return total_result, total_error
   finally:
       sys.stdout = original_stdout
       sys.stderr = original_stderr
       log_file.close()
       os.chdir(current_cwd)
       # Restore environment variables
       if need_cpu_fallback:
           if orig_cuda is None:
               os.environ.pop("CUDA_VISIBLE_DEVICES", None)
           else:
               os.environ["CUDA_VISIBLE_DEVICES"] = orig_cuda
           if orig_jax_x4 is None:
               os.environ.pop("JAX_ENABLE_X64", None)
           else:
               os.environ["JAX_ENABLE_X64"] = orig_jax_x4
           if orig_jax_plat is None:
               os.environ.pop("JAX_PLATFORMS", None)
           else:
               os.environ["JAX_PLATFORMS"] = orig_jax_plat

##DeepSeek V7
#import os
#import numpy as np
#import mdtraj as md
#import freud
#from scipy.ndimage import gaussian_filter1d
#from scipy.optimize import curve_fit
#from collections import Counter
#import re
#
## Make sure 'names' is imported (contains PROJECT_DIR, etc.)
#from files.python_files import names

# ---- Constants ----
N_SIGMA = 3.0

# ---- Helper functions ----
def _gauss(r, A, mu, sigma, c):
    return A * np.exp(-0.5 * ((r - mu) / sigma) ** 2) + c

def _fit_shell(r, g, lo, hi, mu0):
    """Fit one Gaussian + constant offset inside [lo, hi]."""
    mask = (r >= lo) & (r <= hi)
    if np.sum(mask) < 5:
        return None
    rw, gw = r[mask], g[mask]
    A0 = gw.max() - np.median(gw)
    p0 = [A0, mu0, 0.25 * (hi - lo), np.median(gw)]
    bounds = ([0.0, lo, 1e-4, -0.5],
              [10 * A0 + 10, hi, hi - lo, 2.0])
    try:
        popt, _ = curve_fit(_gauss, rw, gw, p0=p0, bounds=bounds, maxfev=100000)
        return popt
    except Exception:
        return None

def _parse_restraint_top(top_path):
    """
    Parse the restraint .top file to extract the ion serial and the list of
    alpha carbon serials (non‑ion atoms that appear in bonds/angles/dihedrals).
    Returns (ion_serial, alpha_carbons_list) or (None, []) if parsing fails.
    """
    import re
    if not os.path.exists(top_path):
        return None, []
    entries = []
    current_section = None
    with open(top_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('['):
                if line == '[ bonds ]':
                    current_section = 'bonds'
                elif line == '[ angles ]':
                    current_section = 'angles'
                elif line == '[ dihedrals ]':
                    current_section = 'dihedrals'
                else:
                    current_section = None
                continue
            if current_section is None:
                continue
            if line.startswith(';'):
                continue
            nums = list(map(int, re.findall(r'\d+', line)))
            if not nums:
                continue
            if current_section == 'bonds' and len(nums) >= 2:
                entries.append(set(nums[:2]))
            elif current_section == 'angles' and len(nums) >= 3:
                entries.append(set(nums[:3]))
            elif current_section == 'dihedrals' and len(nums) >= 4:
                entries.append(set(nums[:4]))
    if not entries:
        return None, []
    # Intersect all entries to find the common atom (the ion)
    common = set.intersection(*entries)
    if len(common) != 1:
        # If more than one common, fallback to most frequent atom
        cnt = Counter()
        for e in entries:
            cnt.update(e)
        ion_serial = cnt.most_common(1)[0][0]
    else:
        ion_serial = list(common)[0]
    # All unique atoms from all entries
    all_atoms = set().union(*entries)
    alpha_carbons = [a for a in all_atoms if a != ion_serial]
    return ion_serial, alpha_carbons

# ---- Main RDF calculation function ----
def calculate_rdf_freud(job, debug_mode=False, target_dir=None):
    try:
        import mdtraj as md
        import freud
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        from scipy.optimize import curve_fit
    except ImportError:
        print("Required libraries for RDF not found.")
        return None, None, None, None, None, None, None, None, None

    with job:
        traj_file = f'{names.NAME_PRO_CANON}.xtc'
        if not os.path.exists(traj_file):
            traj_file = f'{names.NAME_PRO_CANON}.trr'
        top_file = f'{names.NAME_PRO_CANON}.gro'
        if not os.path.exists(top_file):
            top_file = 'init.gro'
        if not os.path.exists(traj_file) or not os.path.exists(top_file):
            return None, None, None, None, None, None, None, None, None

        try:
            traj = md.load(traj_file, top=top_file)
        except Exception as e:
            print(f"Error loading trajectory for job {job.id}: {e}")
            return None, None, None, None, None, None, None, None, None

        metal_name = job.sp.metal
        metal_sel = traj.topology.select(f"name {metal_name} or element {metal_name}")
        if len(metal_sel) == 0:
            print(f"Warning: No metal atoms found for {metal_name}")
            return None, None, None, None, None, None, None, None, None

        # ---- Determine target atoms (oxygen or alpha carbons) ----
        polypeptide = getattr(job.sp, 'polypeptide', None)
        print(f"DEBUG: polypeptide = {polypeptide} (type: {type(polypeptide)})")

        target_sel = None
        use_oxygen = True ######################################################################

        if polypeptide and polypeptide != 'None' and str(polypeptide).strip():
            # Build path to the addendum .top file
            top_path = os.path.join(names.PROJECT_DIR, "files", "addendums",
                                    f"appendFEP_chelate_init_{polypeptide}.top")
            print(f"DEBUG: Looking for restraint file at: {top_path}")
            if not os.path.exists(top_path):
                fallback_path = os.path.join(names.PROJECT_DIR, f"{polypeptide}.top")
                if os.path.exists(fallback_path):
                    top_path = fallback_path
                    print(f"DEBUG: Found fallback file at: {top_path}")
                else:
                    print(f"WARNING: Restraint top file not found. Falling back to oxygen.")
                    use_oxygen = True
            if not use_oxygen and os.path.exists(top_path):
                ion_serial, alpha_serials = _parse_restraint_top(top_path)
                print(f"DEBUG: ion_serial = {ion_serial}, alpha_serials = {alpha_serials}")
                if not alpha_serials:
                    print(f"WARNING: No alpha carbons found in {top_path}. Falling back to oxygen.")
                    use_oxygen = True
                else:
                    # Convert serials (1‑based) to zero‑based indices
                    indices = [a - 1 for a in alpha_serials]
                    sel_str = " or ".join([f"index {i}" for i in indices])
                    print(f"DEBUG: Selection string: {sel_str}")
                    target_sel = traj.topology.select(sel_str)
                    if target_sel is None or len(target_sel) == 0:
                        print(f"WARNING: Could not select alpha carbons from serials {alpha_serials}. Falling back to oxygen.")
                        use_oxygen = True
                    else:
                        print(f"INFO: Selected {len(target_sel)} alpha carbon atoms for RDF analysis.")
        else:
            print("DEBUG: No polypeptide attribute, using oxygen atoms.")
            use_oxygen = True

        if use_oxygen:
            print("INFO: Using oxygen atoms for RDF analysis.")
            target_sel = traj.topology.select("index 167 201 202 118 48 59 168 102 103 74 and element O")
            #target_sel = traj.topology.select("(name O or name OW or element O) and not (resname TIP4P)")
            #target_sel = traj.topology.select("name O or name OW or element O")
            if target_sel is None or len(target_sel) == 0:
                print("ERROR: No oxygen atoms found. Cannot proceed.")
                return None, None, None, None, None, None, None, None, None

        if target_sel is None or len(target_sel) == 0:
            print("ERROR: No target atoms selected. Aborting RDF.")
            return None, None, None, None, None, None, None, None, None

        # ---- Compute RDF ----
        min_box_length = np.min(traj.unitcell_lengths)
        r_max = min_box_length / 2.01
        rdf = freud.density.RDF(bins=200, r_max=r_max, r_min=0.0)
        for i, frame in enumerate(traj.xyz):
            box_matrix = traj.unitcell_vectors[i]
            box = freud.box.Box.from_matrix(box_matrix)
            system = (box, frame[target_sel])
            rdf.compute(system, query_points=frame[metal_sel], reset=False)

        r = rdf.bin_centers
        y = rdf.rdf
        y_smooth = gaussian_filter1d(y, sigma=1.0)

        # ---- Detect extrema (1st, 2nd, 3rd) ----
        mask = (r < 0.5)
        if np.sum(mask) == 0:
            return None, None, None, None, None, None, None, None, None
        peak1_idx = np.argmax(y_smooth[mask])
        max_1 = r[peak1_idx]

        min1_idx = len(r) - 1
        for j in range(peak1_idx, len(r) - 1):
            if y_smooth[j] < y_smooth[j+1]:
                min1_idx = j
                break
        min_1 = r[min1_idx]
        cn_1_old = rdf.n_r[min1_idx]

        # 2nd maximum
        search_range = slice(min1_idx, len(r))
        if np.sum(y_smooth[search_range]) == 0:
            max_2 = 0.0
            peak2_idx = len(r) - 1
        else:
            peak2_idx = min1_idx + np.argmax(y_smooth[search_range])
            max_2 = r[peak2_idx]

        min2_idx = len(r) - 1
        for j in range(peak2_idx, len(r) - 1):
            if y_smooth[j] < y_smooth[j+1]:
                min2_idx = j
                break
        if min2_idx == len(r) - 1 or peak2_idx == len(r) - 1:
            min_2 = 0.0
            cn_2_old = 0.0
        else:
            min_2 = r[min2_idx]
            cn_2_old = rdf.n_r[min2_idx]

        # 3rd maximum
        if min2_idx >= len(r) - 1:
            max_3 = 0.0
            peak3_idx = len(r) - 1
        else:
            search_range2 = slice(min2_idx, len(r))
            if np.sum(y_smooth[search_range2]) == 0:
                max_3 = 0.0
                peak3_idx = len(r) - 1
            else:
                peak3_idx = min2_idx + np.argmax(y_smooth[search_range2])
                max_3 = r[peak3_idx]

        min3_idx = len(r) - 1
        for j in range(peak3_idx, len(r) - 1):
            if y_smooth[j] < y_smooth[j+1]:
                min3_idx = j
                break
        if min3_idx == len(r) - 1 or peak3_idx == len(r) - 1:
            min_3 = 0.0
            cn_3_old = 0.0
        else:
            min_3 = r[min3_idx]
            cn_3_old = rdf.n_r[min3_idx]

        # ---- Gaussian fitting for shell onsets ----
        maxima = [max_1, max_2, max_3]
        minima = [min_1, min_2, min_3]
        valid = [(m, mn) for m, mn in zip(maxima, minima) if m > 0 and mn > 0]
        fits, onsets = [], []
        if len(valid) >= 2:
            lefts = [0.0] + [mn for _, mn in valid[:-1]]
            for i, ((mu0, mn), lo) in enumerate(zip(valid, lefts)):
                hi = mn
                popt = _fit_shell(r, y_smooth, lo, hi, mu0)
                if popt is not None:
                    A, mu, sigma, c = popt
                    fits.append((A, mu, sigma, c))
                    onset = np.clip(mu - N_SIGMA * sigma, lo, mu)
                    onsets.append(onset)
                else:
                    fits.append(None)
                    onsets.append(mn)   # fallback to minimum
            # Compute CNs at onsets
            if len(onsets) >= 2:
                cn_1_new = np.interp(onsets[1], r, rdf.n_r)
            else:
                cn_1_new = cn_1_old
            if len(onsets) >= 3:
                cn_2_new = np.interp(onsets[2], r, rdf.n_r)
            else:
                cn_2_new = cn_2_old
            cn_3_new = 0.0
        else:
            cn_1_new = cn_1_old
            cn_2_new = cn_2_old
            cn_3_new = 0.0

        # ---- Debug plots ----
        if debug_mode and target_dir is not None:
            import matplotlib.pyplot as plt
            import pandas as pd
            debug_txt_path = os.path.join(target_dir, f"{job.id}_debug_rdf.txt")
            df = pd.DataFrame({"r_nm": r, "rdf": y, "rdf_smooth": y_smooth, "cn": rdf.n_r})
            df.to_csv(debug_txt_path, index=False, sep='\t')

            # RDF plot
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(r, y, color='tab:blue', alpha=0.5, label='Raw RDF')
            ax.plot(r, y_smooth, color='tab:red', label='Smoothed RDF')
            ax.axvline(x=min_1, color='r', linestyle='--', label=f'min 1 ({min_1:.3f} nm)')
            ax.axvline(x=max_1, color='orange', linestyle=':', label=f'max 1 ({max_1:.3f} nm)')
            if min_2 > 0:
                ax.axvline(x=min_2, color='g', linestyle='--', label=f'min 2 ({min_2:.3f} nm)')
            if max_2 > 0:
                ax.axvline(x=max_2, color='purple', linestyle=':', label=f'max 2 ({max_2:.3f} nm)')
            if min_3 > 0:
                ax.axvline(x=min_3, color='brown', linestyle='--', label=f'min 3 ({min_3:.3f} nm)')
            if max_3 > 0:
                ax.axvline(x=max_3, color='cyan', linestyle=':', label=f'max 3 ({max_3:.3f} nm)')
            colours = ['tab:blue', 'tab:orange', 'tab:green']
            rr = np.linspace(r.min(), r.max(), 2000)
            for i, (popt, col) in enumerate(zip(fits, colours)):
                if popt is not None:
                    A, mu, sigma, c = popt
                    ax.plot(rr, _gauss(rr, A, mu, sigma, c), color=col, lw=1.2, alpha=0.7,
                            label=f'shell {i+1} fit')
                    onset = onsets[i]
                    ax.axvline(x=onset, color=col, linestyle='-.', lw=1.5,
                               label=f'onset {i+1} ({onset:.3f} nm)')
            ax.legend()
            ax.set_xlabel('Distance r (nm)')
            ax.set_ylabel('RDF g(r)')
            fig.tight_layout()
            plt.savefig(os.path.join(target_dir, f"{job.id}_debug_rdf.png"), dpi=300)
            plt.close(fig)

            # CN plot
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(r, rdf.n_r, color='tab:green', label='CN')
            ax.axvline(x=min_1, color='r', linestyle='--', label=f'min 1')
            if min_2 > 0:
                ax.axvline(x=min_2, color='g', linestyle='--', label=f'min 2')
            if min_3 > 0:
                ax.axvline(x=min_3, color='brown', linestyle='--', label=f'min 3')
            for i, onset in enumerate(onsets):
                if onset > 0:
                    val = np.interp(onset, r, rdf.n_r)
                    ax.plot(onset, val, 'o', color=colours[i], ms=6)
                    ax.annotate(f'CN={val:.2f}', xy=(onset, val),
                                xytext=(5, -10), textcoords='offset points',
                                color=colours[i], fontsize=9)
            ax.set_xlabel('Distance r (nm)')
            ax.set_ylabel('Coordination Number')
            max_cn = np.interp(r[-1], r, rdf.n_r)
            ax.set_ylim(0, max_cn * 1.1)
            ax.legend()
            fig.tight_layout()
            plt.savefig(os.path.join(target_dir, f"{job.id}_debug_cn.png"), dpi=300)
            plt.close(fig)

        # Return: min_1, max_1, cn_1_new, min_2, max_2, cn_2_new, min_3, max_3, cn_3_new
        return min_1, max_1, cn_1_new, min_2, max_2, cn_2_new, min_3, max_3, cn_3_new

def calculate_distance_time(job, target_dir):
    """
    Compute distance from ion to selected atoms as a function of time.
    Concatenates EQ_NVT.trr, EQ_NPT_BERENDSEN.trr, EQ_CANON.trr, PRO_CANON.trr (if present).
    Uses periodic boundary conditions via mdtraj.compute_distances.
    Writes per‑atom distance files and mean files, plus pro_canon_start_time.
    """
    import mdtraj as md
    import numpy as np
    import os

    with job:
        # λ‑index
        bonded = round(job.sp.lambda_BONDED, 5)
        ele = round(job.sp.lambda_ELE, 5)
        lj = round(job.sp.lambda_LJ, 5)
        lambda_idx = names.eleLam_ljLam_to_initLam[(bonded, ele, lj)]
        lambda_str = f"lam{lambda_idx:02d}"

        # Topology
        top_file = f'{names.NAME_PRO_CANON}.gro'
        if not os.path.exists(top_file):
            top_file = 'init.gro'
        if not os.path.exists(top_file):
            print(f"Missing topology for {job.id}")
            return

        # Concatenate segments
        traj_files = [
            f'{names.NAME_EQ_CANON}.trr',
            f'{names.NAME_PRO_CANON}.trr'
        ]

        all_xyz, all_time, all_box = [], [], []
        last_time = 0.0
        pro_canon_start = None

        for i, fname in enumerate(traj_files):
            if not os.path.exists(fname):
                continue
            try:
                seg = md.load(fname, top=top_file)
                if seg.n_frames == 0:
                    continue
                t = seg.time
                dt = t[1] - t[0] if len(t) > 1 else 0.001
                offset = 0.0 if i == 0 else last_time + dt - t[0]
                t_shifted = t + offset
                last_time = t_shifted[-1]
                all_xyz.append(seg.xyz)
                all_time.append(t_shifted)
                all_box.append(seg.unitcell_vectors)   # store box for each frame
                if fname == f'{names.NAME_PRO_CANON}.trr':
                    pro_canon_start = t_shifted[0]
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                continue

        if not all_xyz:
            return

        # Save PRO_CANON start time
        if pro_canon_start is not None:
            with open(os.path.join(target_dir, f"pro_canon_start_time_{lambda_str}.txt"), 'w') as f:
                f.write(f"{pro_canon_start}\n")

        # Concatenate
        xyz = np.concatenate(all_xyz, axis=0)
        time = np.concatenate(all_time, axis=0)
        box = np.concatenate(all_box, axis=0)

        # Create trajectory
        ref_seg = md.load(traj_files[0], top=top_file)
        traj = md.Trajectory(xyz, ref_seg.topology, time=time)
        # Assign box vectors
        traj.unitcell_vectors = box

        # Metal ion
        metal_name = job.sp.metal
        ion_idx = traj.topology.select(f"name {metal_name} or element {metal_name}")
        if len(ion_idx) == 0:
            return
        ion_idx = ion_idx[0]

        # Selections
        polypeptide = getattr(job.sp, 'polypeptide', None)
        selections = {}
        if polypeptide and polypeptide != 'DUM3+':
            from files.python_files.misc_funct import _parse_restraint_top
            top_path = os.path.join(names.PROJECT_DIR, "files", "addendums",
                                    f"appendFEP_chelate_init_{polypeptide}.top")
            ion_serial, alpha_serials = _parse_restraint_top(top_path)
            if alpha_serials:
                selections['CA'] = ([s-1 for s in alpha_serials], alpha_serials)
            else:
                ca = traj.topology.select("name CA")
                if len(ca) > 0:
                    selections['CA'] = (ca.tolist(), [i+1 for i in ca])

            o_file = os.path.join(names.PROJECT_DIR, "files", "addendums", "charges",
                                  f"custom_charges_{polypeptide}_Oxygen.top")
            if os.path.exists(o_file):
                o_serials = []
                with open(o_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith(';'):
                            parts = line.split()
                            if parts:
                                try:
                                    o_serials.append(int(parts[0]))
                                except ValueError:
                                    pass
                if o_serials:
                    selections['O_coord'] = ([s-1 for s in o_serials], o_serials)

        # OHN
        ohn = traj.topology.select("element O or element N or element H")
        ohn = ohn[ohn != ion_idx]
        if len(ohn) > 0:
            selections['OHN'] = (ohn.tolist(), [])

        # Compute and write data with PBC
                # Compute and write data with PBC
        for label, (indices, serials) in selections.items():
            if len(indices) == 0:
                continue

            # Mean distance using all pairs (ion vs each atom)
            pairs = [[ion_idx, idx] for idx in indices]
            all_dists = md.compute_distances(traj, pairs, periodic=True)
            mean_dists = np.mean(all_dists, axis=1)
            np.savetxt(os.path.join(target_dir, f"distance_{label}_{lambda_str}.txt"),
                       np.column_stack((time, mean_dists)),
                       header="time (ps)\tmean_distance (nm)", comments='')

            # Per‑atom distances
            if serials and len(serials) == len(indices):
                for idx, serial in zip(indices, serials):
                    dists = md.compute_distances(traj, [[ion_idx, idx]], periodic=True)[:, 0]
                    np.savetxt(os.path.join(target_dir, f"distance_{label}_serial{serial}_{lambda_str}.txt"),
                               np.column_stack((time, dists)),
                               header="time (ps)\tdistance (nm)", comments='')

        # --- MARKER CREATION (added here) ---
        # Create marker file to indicate data is written
        marker_file = os.path.join(target_dir, f"distance_data_{lambda_str}.done")
        with open(marker_file, 'w') as f:
            f.write("Data written successfully\n")

        return

# ==============================================================================
# RESTRAINT OXYGEN MAPPINGS PER POLYPEPTIDE
# ==============================================================================
POLYPEPTIDE_RESTRAINT_MAP = {
    "LBT5-": {
        "bond_oxygens": [119],
        "angle_oxygens": [202, 203], #[75, 202, 203],
        "dihedral_set1": [49, 50, 103, 104],
        "dihedral_set2": [168, 169],
    },
    "LBT3-": {
        "bond_oxygens": [],
        "angle_oxygens": [],
        "dihedral_set1": [],
        "dihedral_set2": [],
    }
}


def calculate_restraint_oxygen_distances(job, target_dir):
    """
    Computes time-series distances from dynamically detected metal ion to each 
    restraint oxygen across EQ_CANON + PRO_CANON trajectories.
    """
    import mdtraj as md
    import numpy as np
    import os

    polypeptide = getattr(job.sp, 'polypeptide', None)
    if polypeptide not in POLYPEPTIDE_RESTRAINT_MAP:
        return

    mapping = POLYPEPTIDE_RESTRAINT_MAP[polypeptide]
    
    # Lambda filter (Include 0.0 for test run, or expand to [0.0, 0.25, 0.5, 0.75, 0.9])
    bonded = round(job.sp.lambda_BONDED, 5)
    ele = round(job.sp.lambda_ELE, 5)
    lj = round(job.sp.lambda_LJ, 5)
    
    target_eles = [0.0]  # Set to [0.0, 0.25, 0.5, 0.75, 0.9] for full runs
    if bonded != 1.0 or lj != 0.0 or not any(np.isclose(ele, t, atol=1e-4) for t in target_eles):
        return

    lambda_idx = names.eleLam_ljLam_to_initLam[(bonded, ele, lj)]
    lambda_str = f"lam{lambda_idx:02d}"

    with job:
        top_file = f'{names.NAME_PRO_CANON}.gro'
        if not os.path.exists(top_file):
            top_file = 'init.gro'
        if not os.path.exists(top_file):
            return

        traj_files = [
            f'{names.NAME_EQ_CANON}.trr',
            f'{names.NAME_PRO_CANON}.trr'
        ]

        all_xyz, all_time, all_box = [], [], []
        last_time = 0.0
        pro_canon_start = None

        for i, fname in enumerate(traj_files):
            if not os.path.exists(fname):
                continue
            try:
                seg = md.load(fname, top=top_file)
                if seg.n_frames == 0:
                    continue
                t = seg.time
                dt = t[1] - t[0] if len(t) > 1 else 0.001
                offset = 0.0 if i == 0 else last_time + dt - t[0]
                t_shifted = t + offset
                last_time = t_shifted[-1]
                all_xyz.append(seg.xyz)
                all_time.append(t_shifted)
                all_box.append(seg.unitcell_vectors)
                if fname == f'{names.NAME_PRO_CANON}.trr':
                    pro_canon_start = t_shifted[0]
            except Exception as e:
                print(f"Error loading {fname} in job {job.id}: {e}")
                continue

        if not all_xyz:
            return

        if pro_canon_start is not None:
            with open(os.path.join(target_dir, f"pro_canon_start_time_{lambda_str}.txt"), 'w') as f:
                f.write(f"{pro_canon_start}\n")

        xyz = np.concatenate(all_xyz, axis=0)
        time = np.concatenate(all_time, axis=0)
        box = np.concatenate(all_box, axis=0)

        ref_seg = md.load(traj_files[0], top=top_file)
        traj = md.Trajectory(xyz, ref_seg.topology, time=time)
        traj.unitcell_vectors = box

        metal_name = job.sp.metal
        ion_sel = traj.topology.select(f"name {metal_name} or resname {metal_name} or element {metal_name}")
        if len(ion_sel) == 0:
            return
        ion_idx = ion_sel[0]

        all_o_serials = (
            mapping["bond_oxygens"] +
            mapping["angle_oxygens"] +
            mapping["dihedral_set1"] +
            mapping["dihedral_set2"]
        )

        for serial in all_o_serials:
            idx = serial - 1
            dists = md.compute_distances(traj, [[ion_idx, idx]], periodic=True)[:, 0]
            
            np.savetxt(
                os.path.join(target_dir, f"restraint_O_serial{serial}_{lambda_str}.txt"),
                np.column_stack((time, dists)),
                header="time(ps)\tdistance(nm)",
                comments=''
            )

        marker_file = os.path.join(target_dir, f"restraint_oxygens_{lambda_str}.done")
        with open(marker_file, 'w') as f:
            f.write("Data written successfully\n")