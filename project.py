import math
import mbuild
import numpy as np
import signac
from flow import FlowProject
from flow.environment import DefaultSlurmEnvironment
import os
import shutil
import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import io
from jinja2 import Environment, FileSystemLoader

PROJECT_DIR = os.path.abspath('.')
INIT_FILE_LIST = ['init.gro','init.top']
GROMACS_PREFIX = 'gmx' # "/usr/local/gromacs/bin/gmx"
#names
NAME_EM = "em"
NAME_EQ_NVT = "EQ_NVT"
NAME_EQ_NPT_BERENDSEN = "EQ_NPT_BERENDSEN"
NAME_EQ_CANON = "EQ_CANON"
NAME_PRO_CANON = "PRO_CANON"
TEMPERATURE = 298.0
PRESSURE = 1.0

MDP_FILE_LIST = [f'{NAME_EQ_NVT}.mdp', f'{NAME_EQ_NPT_BERENDSEN}.mdp', f'{NAME_EQ_CANON}.mdp', f'{NAME_PRO_CANON}.mdp']

ljLam_eleLam_to_initLam = {
    (0.0,	    0.00):	    0,
    (0.2,	    0.00):	    1,
    (0.4,	    0.00):	    2,
    (0.6,	    0.00):	    3,
    (0.8,	    0.00):	    4,
    (0.9,	    0.00):	    5,
    (1.0,	    0.00):	    6,
    (1.0,	    0.20):	    7,
    (1.0,	    0.40):	    8,
    (1.0,	    0.55):      9,
    (1.0,	    0.60):	    10,
    (1.0,	    0.65):   	11,
    (1.0,	    0.70):	    12,
    (1.0,	    0.75):      13,
    (1.0,	    0.80):	    14,
    (1.0,	    0.90):	    15,
    (1.0,	    1.00):	    16
}

METAL_FORMAL_CHARGES = {
    'Al': 3, 'Fe': 3, 'Cr': 3, 'In': 3, 'Tl': 3,
    'Y': 3, 'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3,
    'Sm': 3, 'Eu': 3, 'Gd': 3, 'Tb': 3, 'Dy': 3,
    'Er': 3, 'Tm': 3, 'Lu': 3,
    'Hf': 4, 'Zr': 4, 'U': 4, 'Pu': 4, 'Th': 4,
}

MIN_CORES = 1
MID_CORES = 6

SMALL_EQ_STEPS      = int(1000000)
MID_EQ_STEPS        = int(2000000)
LONG_EQ_STEPS       = int(4000000)
SLOW_OUTPUT         = int(20000)
NORMAL_CALC         = int(100)

PRO_STEPS             = int(500000)
PRO_FREE_ENERGY_STEPS = int(500000)
FAST_OUTPUT           = int(1000)
FAST_CALC             = int(100)

MIN_HOURS = 2.0; MID_HOURS = 8.0; DAY_WAIT = 24.0; TWO_DAYS = 48.0
RCUT = 1.4
GENERAL_LOCAL_DATA = 'raw_general_data_for'
GENERAL_GLOBAL_DATA = 'aggregate_general_Data'

@FlowProject.label
def init_written(job):
    with(job):
        test_passed = False
        for i in INIT_FILE_LIST:
            if job.isfile(i):
                test_passed = True
            else:
                test_passed = False
                break

    return test_passed

@FlowProject.label
def mdp_written(job):
    with(job):
        test_passed = False
        for i in MDP_FILE_LIST:
            if job.isfile(i):
                test_passed = True
            else:
                test_passed = False
                break

    return test_passed

@FlowProject.label
def select_metals(job):
    with(job):
        test_passed = False
        metals_to_run = ['Fe','Gd','Hf']
        for metal in metals_to_run:
            if job.sp.metal == metal:
                test_passed = True
                break

    return test_passed

@FlowProject.label
def eq_nvt_post(job):
    with(job):
        test_passed = False
        if job.isfile(f"{NAME_EQ_NVT}.log"):
            file_with_lines = open(f"{NAME_EQ_NVT}.log","r")
            lines = file_with_lines.readlines()
            file_with_lines.close()
            for single_line in lines:
                if "Received the " in single_line:
                    if " signal, stopping within" in single_line:
                        test_passed = False
                        break

                if "Finished mdrun on " in single_line:
                    test_passed = True
                    break

    return test_passed

@FlowProject.label
def eq_npt_post_beren(job):
    with(job):
        test_passed = False
        if job.isfile(f"{NAME_EQ_NPT_BERENDSEN}.log"):
            file_with_lines = open(f"{NAME_EQ_NPT_BERENDSEN}.log","r")
            lines = file_with_lines.readlines()
            file_with_lines.close()
            for single_line in lines:
                if "Received the " in single_line:
                    if " signal, stopping within" in single_line:
                        test_passed = False
                        break

                if "Finished mdrun on " in single_line:
                    test_passed = True
                    break

    return test_passed

@FlowProject.label
def eq_canon_post(job):
    with(job):
        test_passed = False
        if job.isfile(f"{NAME_EQ_CANON}.log"):
            file_with_lines = open(f"{NAME_EQ_CANON}.log","r")
            lines = file_with_lines.readlines()
            file_with_lines.close()
            for single_line in lines:
                if "Received the " in single_line:
                    if " signal, stopping within" in single_line:
                        test_passed = False
                        break

                if "Finished mdrun on " in single_line:
                    test_passed = True
                    break

    return test_passed

@FlowProject.label
def pro_canon_post(job):
    with(job):
        test_passed = False
        if job.isfile(f"{NAME_PRO_CANON}.log"):
            file_with_lines = open(f"{NAME_PRO_CANON}.log","r")
            lines = file_with_lines.readlines()
            file_with_lines.close()
            for single_line in lines:
                if "Received the " in single_line:
                    if " signal, stopping within" in single_line:
                        test_passed = False
                        break

                if "Finished mdrun on " in single_line:
                    test_passed = True
                    break

    return test_passed

@FlowProject.label
def free_energy_bar_copied(job):
    with(job):
        test_passed = False
        if job.isfile(f'{NAME_PRO_CANON}.xvg'):
            current_lambda = ljLam_eleLam_to_initLam[round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]
            if job.isfile(f'{NAME_PRO_CANON}_{current_lambda}.xvg'):
                test_passed = True

    return test_passed


@FlowProject.label
def data_collected(job):
    test_passed = False
    local_name_of_file = f'{GENERAL_GLOBAL_DATA}.txt'
    if os.path.exists(local_name_of_file):
        with open(local_name_of_file, "r") as f:
            contents = f.read()
            if job.id in contents:
                test_passed = True

    return test_passed


def simple_mdp_writer(job,mdp_name,parameters,constraints=None,templates_dir=None,template_name=None):
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
            'lincs_order' : '6'
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

current_directory = os.getcwd()
current_directory_name = os.path.basename(current_directory)
project = signac.get_project()

class Custom_environment(DefaultSlurmEnvironment):
    template = "v3_2025_gpu_potoff.sh"

@FlowProject.post(init_written)
@FlowProject.post(mdp_written)
@FlowProject.operation(directives={ "np": int(1),  "ngpu": 0, "memory": 3.2, "walltime": MIN_HOURS})
def build_input(job):
    with(job):
        import sys
        from rdkit import Chem
        from openff.toolkit import Molecule, Topology, ForceField
        from openff.units import unit
        import mbuild.packing

        conda_bin = os.path.dirname(sys.executable)
        os.environ['PATH'] = f'{conda_bin}:{os.environ.get("PATH", "")}'
        mbuild.packing.PACKMOL = shutil.which('packmol')

        counterion_count = 4 if job.sp.metal in ('U', 'Hf') else 3

        tip3p_mb = mbuild.load(f'{PROJECT_DIR}/files/coordinates/TIP3P.mol2')
        tip3p_mb.name = 'WAT'
        cation_mb = mbuild.load(f'{PROJECT_DIR}/files/coordinates/metal_cations/{job.sp.metal}.mol2')
        cation_mb.name = job.sp.metal
        cl_mb = mbuild.load(f'{PROJECT_DIR}/files/coordinates/neutralizing_anions/Cl.mol2')
        cl_mb.name = 'Cl'

        starting_box = mbuild.fill_box(
            compound=[tip3p_mb, cation_mb, cl_mb],
            n_compounds=[1000, 1, counterion_count],
            box=[3.8, 3.8, 3.8]
        )

        water_mol = Molecule.from_smiles('O')
        cl_mol = Molecule.from_smiles('[Cl-]')
        cation_rd = Chem.MolFromMol2File(
            f'{PROJECT_DIR}/files/coordinates/metal_cations/{job.sp.metal}.mol2', removeHs=False
        )
        cation_mol = Molecule.from_rdkit(cation_rd)
        cation_mol.atoms[0].formal_charge = METAL_FORMAL_CHARGES[job.sp.metal] * unit.elementary_charge

        mols = [water_mol] * 1000 + [cation_mol] + [cl_mol] * counterion_count
        topology = Topology.from_molecules(mols)
        topology.box_vectors = np.eye(3) * 3.8 * unit.nanometer

        ff = ForceField('tip3p.offxml', f'{PROJECT_DIR}/files/xml/custom_ree.offxml')
        interchange = ff.create_interchange(topology)
        interchange.positions = starting_box.xyz * unit.nanometer
        interchange.to_gromacs(prefix='init')

        if os.path.exists('init_pointenergy.mdp'):
            os.remove('init_pointenergy.mdp')

    current_lambda = ljLam_eleLam_to_initLam[round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]

    parameters = {
        "integrator": "sd",
        "dt": 0.001,
        "nsteps": MID_EQ_STEPS,
        "output_control": SLOW_OUTPUT,
        "nstcalcenergy": NORMAL_CALC,
        "rcoulomb": RCUT,
        "coulombtype": "PME",
        "coulomb_modifier": "None",
        "rcoulomb_switch": 0.0,
        "vdwtype": "Cut-off",
        "vdw_modifier": "None",
        "rvdw": RCUT,
        "rvdw_switch": 0.0,
        "DispCorr": "EnerPres",
        "tcoupl": "no",
        "ref_t": TEMPERATURE
    }

    simple_mdp_writer(job,mdp_name=f'{NAME_EQ_NVT}.mdp',parameters=parameters, constraints=None,templates_dir=f'{PROJECT_DIR}/files/mdp/',template_name='NVT_template.mdp')

    parameters.update({
        "nsteps": LONG_EQ_STEPS,
        "output_control": SLOW_OUTPUT,
        "nstcalcenergy": NORMAL_CALC,
        "ref_t": TEMPERATURE,
        "pcoupl": "Berendsen",
        "ref_p": PRESSURE,
        "compressibility": 4.5e-5
    })

    simple_mdp_writer(job,mdp_name=f'{NAME_EQ_NPT_BERENDSEN}.mdp',parameters=parameters, constraints=None,templates_dir=f'{PROJECT_DIR}/files/mdp/',template_name='NPTmdp_template.mdp')

    parameters.update({
        "nsteps": SMALL_EQ_STEPS,
        "output_control": SLOW_OUTPUT,
        "nstcalcenergy": NORMAL_CALC,
        "current_lambda": current_lambda,
        "molecule_of_interest": job.sp.metal,
        "nstdhdl": int(NORMAL_CALC*10)
    })

    simple_mdp_writer(job,mdp_name=f'{NAME_EQ_CANON}.mdp',parameters=parameters, constraints=None,templates_dir=f'{PROJECT_DIR}/files/mdp/',template_name='free_energy_NPTmdp_template.mdp')

    parameters.update({
        "nsteps": PRO_FREE_ENERGY_STEPS,
        "output_control": FAST_OUTPUT,
        "nstcalcenergy": NORMAL_CALC,
        "current_lambda": current_lambda,
        "molecule_of_interest": job.sp.metal,
        "nstdhdl": int(NORMAL_CALC*10)
    })

    simple_mdp_writer(job,mdp_name=f'{NAME_PRO_CANON}.mdp',parameters=parameters, constraints=None,templates_dir=f'{PROJECT_DIR}/files/mdp/',template_name='free_energy_NPTmdp_template.mdp')

@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.post(eq_nvt_post)
@FlowProject.operation(directives={"np": int(MID_CORES),  "ngpu": 1, "memory": 3.2, "walltime": MID_HOURS},with_job=True,cmd=True)
def EQ_NVT(job):
    build_mdp = str(f'{GROMACS_PREFIX} grompp -f {NAME_EQ_NVT}.mdp -c init.gro -p init.top -o {NAME_EQ_NVT}.tpr -maxwarn 999')
    run_gmx = str(f'{GROMACS_PREFIX} mdrun -nt {MID_CORES} -deffnm {NAME_EQ_NVT}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')

    return run_command

@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(eq_nvt_post)
@FlowProject.post(eq_npt_post_beren)
@FlowProject.operation(directives={"np": int(MID_CORES),  "ngpu": 1, "memory": 3.2, "walltime": TWO_DAYS},with_job=True,cmd=True)
def EQ_NPT_BERENDSEN(job):
    build_mdp = str(f'{GROMACS_PREFIX} grompp -f {NAME_EQ_NPT_BERENDSEN}.mdp -c {NAME_EQ_NVT}.gro -p init.top -o {NAME_EQ_NPT_BERENDSEN}.tpr -maxwarn 999')
    run_gmx = str(f'{GROMACS_PREFIX} mdrun -nt {MID_CORES} -deffnm {NAME_EQ_NPT_BERENDSEN}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')

    return run_command

@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(eq_npt_post_beren)
@FlowProject.post(eq_canon_post)
@FlowProject.operation(directives={"np": int(MID_CORES),  "ngpu": 1, "memory": 3.2, "walltime": TWO_DAYS},with_job=True,cmd=True)
def EQ_CANON(job):
    build_mdp = str(f'{GROMACS_PREFIX} grompp -f {NAME_EQ_CANON}.mdp -c {NAME_EQ_NPT_BERENDSEN}.gro -p init.top -o {NAME_EQ_CANON}.tpr -maxwarn 999')
    run_gmx = str(f'{GROMACS_PREFIX} mdrun -nt {MID_CORES} -deffnm {NAME_EQ_CANON}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')

    return run_command

@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(eq_canon_post)
@FlowProject.post(pro_canon_post)
@FlowProject.operation(directives={"np": int(MID_CORES),  "ngpu": 1, "memory": 3.2, "walltime": TWO_DAYS},with_job=True,cmd=True)
def PRO_CANON(job):
    build_mdp = str(f'{GROMACS_PREFIX} grompp -f {NAME_PRO_CANON}.mdp -c {NAME_EQ_CANON}.gro -p init.top -o {NAME_PRO_CANON}.tpr -maxwarn 999')
    run_gmx = str(f'{GROMACS_PREFIX} mdrun -nt {MID_CORES} -deffnm {NAME_PRO_CANON}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')

    return run_command


@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(pro_canon_post)
@FlowProject.post(free_energy_bar_copied)
@FlowProject.operation(directives={"np": int(MID_CORES),  "ngpu": 0, "memory": 3.2, "walltime": MIN_HOURS},with_job=True,cmd=True)
def FREE_ENERGY_FILES_RENAMED(job):
    current_lambda = ljLam_eleLam_to_initLam[round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]
    run_command = str(f'cp {NAME_PRO_CANON}.xvg {NAME_PRO_CANON}_{current_lambda}.xvg')

    return run_command


@FlowProject.pre(data_collected) ## DUMMY TO AVOID THE JOB
@FlowProject.pre(free_energy_bar_copied)
@FlowProject.pre(pro_canon_post)
@FlowProject.post(data_collected)
@FlowProject.operation(directives={ "np": int(1),  "ngpu": 0, "memory": 1.1, "walltime": MIN_HOURS})
def GRAPH_AND_COLLECT_PROPERTIES(job):
    with(job):
        properties_of_interest = ["Potential", "Pressure", "Total-Energy", "Temperature", "Density"]
        properties_of_interest_to_search_string_dict = {
            properties_of_interest[0] : ['Potential','(kJ/mol)'],
            properties_of_interest[1] : ['Pressure','(bar)'],
            properties_of_interest[2] : ['Total Energy','(kJ/mol)'],
            properties_of_interest[3] : ['Temperature','(K)'],
            properties_of_interest[4] : ['Density','(kg/m^3)']
        }

        properties_of_interest_storage_dict = {
            properties_of_interest[0] : 0.0,
            properties_of_interest[1] : 0.0,
            properties_of_interest[2] : 0.0,
            properties_of_interest[3] : 0.0,
            properties_of_interest[4] : 0.0
        }

        gromacs_input = b'1\n0\n'
        result = subprocess.run(
        [f"{GROMACS_PREFIX}", "energy", "-f", f"{NAME_PRO_CANON}.edr", "-o", "dummy_data.xvg"],
        input=gromacs_input,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
        )

        with open("gmx_energy_index_reader.txt", "wb") as f:
            f.write(result.stdout)

        with open("gmx_energy_index_reader.txt", "r") as f:
            text = f.read()

        pattern = r'\b(\d+)\s+([^\s][^ \n]*)'
        matches = re.findall(pattern, text, re.MULTILINE)

        index_map = {name.strip(): int(index) for index, name in matches}
        results = {prop: index_map[prop] for prop in properties_of_interest if prop in index_map}

        newline_string = "\n".join(str(results[prop]) for prop in properties_of_interest if prop in results)

        p = subprocess.Popen([f'{GROMACS_PREFIX}', '-quiet', 'energy', '-f', f'{NAME_PRO_CANON}.edr', '-o', f'{GENERAL_LOCAL_DATA}_{NAME_PRO_CANON}.xvg'], stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        out,err = p.communicate(f'{newline_string}'.encode('utf-8'))
        capture = out.decode()

        Dummy_GMX_output = open(f'{GENERAL_LOCAL_DATA}_{NAME_PRO_CANON}.txt','w')
        Dummy_GMX_output.write(capture)
        Dummy_GMX_output.close()

        Dummy_GMX_output = open(f'{GENERAL_LOCAL_DATA}_{NAME_PRO_CANON}.txt','r')
        aggregate_surTenFile = open(f"../../{GENERAL_GLOBAL_DATA}.txt",'a')

        for a_single_line in Dummy_GMX_output:
            for property_str in properties_of_interest:
                search_property_str_dict = properties_of_interest_to_search_string_dict[property_str]
                search_str_start = search_property_str_dict[0]
                search_str_end = search_property_str_dict[1]

                if (search_str_start in a_single_line) and (search_str_end in a_single_line):
                    numpyCatcher=np.fromstring(a_single_line.strip(f'{search_str_start}{search_str_end}'),dtype=float,sep=' ')[0]
                    properties_of_interest_storage_dict[property_str] = numpyCatcher

        aggregate_surTenFile.write(f"{job.id:<42} {job.sp.lambda_LJ:<8} {job.sp.lambda_ELE:<8} {properties_of_interest_storage_dict['Potential']:<42} "
                                  f" {properties_of_interest_storage_dict['Pressure']:<42} "
                                  f" {properties_of_interest_storage_dict['Total-Energy']:<42} "
                                  f" {properties_of_interest_storage_dict['Temperature']:<42} "
                                  f" {properties_of_interest_storage_dict['Density']:<42} \n")

        Dummy_GMX_output.close(); aggregate_surTenFile.close()

        xvg_png_datasource = open(f'{GENERAL_LOCAL_DATA}_{NAME_PRO_CANON}.xvg','r')
        lines = xvg_png_datasource.readlines()

        header_lines = []
        data_lines = []

        for line in lines:
            if line.startswith('@') or line.startswith('#'):
                header_lines.append(line)
            else:
                data_lines.append(line.strip())

        column_names = {}
        xaxis_label = "Time (ps)"
        yaxis_label = ""

        for line in header_lines:
            if line.startswith('@ s'):
                match = re.search(r'@ s(\d+) legend "(.+)"', line)
                if match:
                    column_names[int(match.group(1))] = match.group(2)
            elif line.startswith('@ xaxis'):
                match = re.search(r'@ xaxis\s+label "(.+)"', line)
                if match:
                    xaxis_label = match.group(1)
            elif line.startswith('@ yaxis'):
                match = re.search(r'@ yaxis\s+label "(.+)"', line)
                if match:
                    yaxis_label = match.group(1)

        ordered_column_names = [xaxis_label]
        for i in range(max(column_names.keys()) + 1 if column_names else 0):
            if i in column_names:
                ordered_column_names.append(column_names[i])

        df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=r'\s+', header=None)
        df.columns = ordered_column_names[:len(df.columns)]

        num_cols = len(df.columns) - 1
        fig, axes = plt.subplots(num_cols, 1, figsize=(10, 5 * num_cols), sharex=True)

        if num_cols == 1:
            axes = [axes]

        for i, col_name in enumerate(df.columns[1:]):
            axes[i].plot(df[xaxis_label], df[col_name])
            axes[i].set_ylabel(f'{col_name} {yaxis_label}')
            axes[i].grid(True)
            key_to_mean_data = ''
            for key, value_list in properties_of_interest_to_search_string_dict.items():
                if col_name in value_list[0]:
                    key_to_mean_data = key
            axes[i].set_title(f'{col_name}; mean {properties_of_interest_storage_dict[key_to_mean_data]}')

        axes[-1].set_xlabel(xaxis_label)
        plt.tight_layout()
        plt.savefig(f'{GENERAL_LOCAL_DATA}_{NAME_PRO_CANON}.png')
        plt.close()


if __name__ == '__main__':
    FlowProject().main()
