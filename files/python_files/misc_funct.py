#from _typeshed import _type_checker_internals
import os
import math
import shutil
import subprocess
from collections import defaultdict
from jinja2 import Environment, FileSystemLoader
import numpy as np
#import parmed
#from parmed import gromacs
#import mbuild as mb
#from foyer import Forcefield
#import foyer
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

# GMXPrefix mapping
GMX_PREFIX = '/usr/local/gromacs/bin/gmx' # for potoff cluster

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
        subprocess.run((f'{GMX_PREFIX}') + str(' density -f ') + str(f'{trr_or_gro}') + str(' -n ') + str(f'{index_file}') + str(' -s ') + str(f'{tpr_file}') + str(' -o ') + str(f'{output_xvg_name}') + str(' -sl ') + str(f'{slices}'), shell=True)


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

#DeepSeek RDF analysis V3
def calculate_rdf_freud(job, debug_mode=False, target_dir=None):
    try:
        import mdtraj as md
        import freud
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        print("Required libraries (mdtraj, freud, scipy) for RDF not found.")
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
        oxygen_sel = traj.topology.select("name O or name OW or element O")
        
        if len(metal_sel) == 0 or len(oxygen_sel) == 0:
            return None, None, None, None, None, None, None, None, None
            
        min_box_length = np.min(traj.unitcell_lengths)
        r_max = min_box_length / 2.01
        rdf = freud.density.RDF(bins=200, r_max=r_max, r_min=0.0)
        
        for i, frame in enumerate(traj.xyz):
            box_matrix = traj.unitcell_vectors[i]
            box = freud.box.Box.from_matrix(box_matrix)
            system = (box, frame[oxygen_sel])
            rdf.compute(system, query_points=frame[metal_sel], reset=False)
            
        r = rdf.bin_centers
        y = rdf.rdf
        
        # Smooth the RDF
        y_smooth = gaussian_filter1d(y, sigma=1.0)
        
        # ---- 1st maximum (within 0.5 nm) ----
        mask = (r < 0.5)
        if np.sum(mask) == 0:
            return None, None, None, None, None, None, None, None, None
        peak1_idx = np.argmax(y_smooth[mask])
        max_1 = r[peak1_idx]
        
        # ---- 1st minimum ----
        min1_idx = len(r) - 1
        for j in range(peak1_idx, len(r) - 1):
            if y_smooth[j] < y_smooth[j+1]:
                min1_idx = j
                break
        min_1 = r[min1_idx]
        cn_1 = rdf.n_r[min1_idx]
        
        # ---- 2nd maximum (global max after min1) ----
        search_range = slice(min1_idx, len(r))
        if np.sum(y_smooth[search_range]) == 0:
            max_2 = 0.0
            peak2_idx = len(r) - 1
        else:
            peak2_idx = min1_idx + np.argmax(y_smooth[search_range])
            max_2 = r[peak2_idx]
        
        # ---- 2nd minimum ----
        min2_idx = len(r) - 1
        for j in range(peak2_idx, len(r) - 1):
            if y_smooth[j] < y_smooth[j+1]:
                min2_idx = j
                break
        if min2_idx == len(r) - 1 or peak2_idx == len(r) - 1:
            min_2 = 0.0
            cn_2 = 0.0
        else:
            min_2 = r[min2_idx]
            cn_2 = rdf.n_r[min2_idx]
        
        # ---- 3rd maximum (global max after min2) ----
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
        
        # ---- 3rd minimum ----
        min3_idx = len(r) - 1
        for j in range(peak3_idx, len(r) - 1):
            if y_smooth[j] < y_smooth[j+1]:
                min3_idx = j
                break
        if min3_idx == len(r) - 1 or peak3_idx == len(r) - 1:
            min_3 = 0.0
            cn_3 = 0.0
        else:
            min_3 = r[min3_idx]
            cn_3 = rdf.n_r[min3_idx]
        
        # ---- Debug plots ----
        if debug_mode and target_dir is not None:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            debug_txt_path = os.path.join(target_dir, f"{job.id}_debug_rdf.txt")
            df = pd.DataFrame({"r_nm": r, "rdf": y, "rdf_smooth": y_smooth, "cn": rdf.n_r})
            df.to_csv(debug_txt_path, index=False, sep='\t')
            
            debug_plot_path_rdf = os.path.join(target_dir, f"{job.id}_debug_rdf.png")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlabel('Distance r (nm)')
            ax.set_ylabel('RDF g(r)')
            ax.plot(r, y, color='tab:blue', alpha=0.5, label='Raw RDF')
            ax.plot(r, y_smooth, color='tab:red', label='Smoothed RDF')
            # Mark minima and maxima
            ax.axvline(x=min_1, color='r', linestyle='--', label=f'min 1 ({min_1:.3f} nm)')
            ax.axvline(x=max_1, color='orange', linestyle=':', label=f'max 1 ({max_1:.3f} nm)')
            if min_2 > 0.0:
                ax.axvline(x=min_2, color='g', linestyle='--', label=f'min 2 ({min_2:.3f} nm)')
            if max_2 > 0.0:
                ax.axvline(x=max_2, color='purple', linestyle=':', label=f'max 2 ({max_2:.3f} nm)')
            if min_3 > 0.0:
                ax.axvline(x=min_3, color='brown', linestyle='--', label=f'min 3 ({min_3:.3f} nm)')
            if max_3 > 0.0:
                ax.axvline(x=max_3, color='cyan', linestyle=':', label=f'max 3 ({max_3:.3f} nm)')
            ax.legend()
            fig.tight_layout()
            plt.savefig(debug_plot_path_rdf, dpi=300)
            plt.close(fig)
            
            debug_plot_path_cn = os.path.join(target_dir, f"{job.id}_debug_cn.png")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlabel('Distance r (nm)')
            ax.set_ylabel('Coordination Number')
            ax.plot(r, rdf.n_r, color='tab:green', label='CN')
            ax.axvline(x=min_1, color='r', linestyle='--', label=f'CN 1: {cn_1:.2f}')
            if min_2 > 0.0:
                ax.axvline(x=min_2, color='g', linestyle='--', label=f'CN 2: {cn_2:.2f}')
            if min_3 > 0.0:
                ax.axvline(x=min_3, color='brown', linestyle='--', label=f'CN 3: {cn_3:.2f}')
            # Set y‑limit based on the largest CN found
            max_cn = max(cn_1, cn_2, cn_3)
            if max_cn > 0:
                ax.set_ylim(0, max_cn * 1.5)
            ax.legend()
            fig.tight_layout()
            plt.savefig(debug_plot_path_cn, dpi=300)
            plt.close(fig)
            
        return min_1, max_1, cn_1, min_2, max_2, cn_2, min_3, max_3, cn_3
