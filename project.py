"""
REE_HYDRA signac workflow project.

Rare earth element hydration free energy simulations using GROMACS
with signac workflow management.

Contributors:
  - dyukovsm <go0719@wayne.edu> (Lead Developer)
  - Gemini (Google DeepMind) (Co-Author)
  - Claude Opus 4.5 (Anthropic) (Co-Author)
"""

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

from files.python_files import names
from files.python_files import misc_funct
from files.python_files.job_tester import (
    init_written,
    mdp_written,
    select_metals,
    eq_nvt_post,
    eq_npt_post_beren,
    eq_canon_post,
    pro_canon_post,
    free_energy_bar_copied,
    data_collected
)

GROMACS_PREFIX = 'gmx' # "/usr/local/gromacs/bin/gmx"

# Cores configuration
BUILD_CORES = 1
SIM_CORES = 6
ANA_CORES = 1

# Walltimes configuration
MIN_HOURS = 2.0
MID_HOURS = 8.0
DAY_WAIT = 24.0
TWO_DAYS = 48.0


project = signac.get_project()


class Custom_environment(DefaultSlurmEnvironment):
    template = "v3_2025_gpu_potoff.sh"


@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.post(eq_nvt_post)
@FlowProject.operation(directives={"np": int(BUILD_CORES), "ngpu": 0, "memory": 3.2, "walltime": MIN_HOURS})
def build_input(job):
    with job:
        import sys
        from rdkit import Chem
        from openff.toolkit import Molecule, Topology, ForceField
        from openff.units import unit
        import mbuild.packing

        conda_bin = os.path.dirname(sys.executable)
        os.environ['PATH'] = f'{conda_bin}:{os.environ.get("PATH", "")}'
        mbuild.packing.PACKMOL = shutil.which('packmol')

        counterion_count = 4 if job.sp.metal in ('U', 'Hf') else 3

        tip3p_mb = mbuild.load(f'{names.PROJECT_DIR}/files/coordinates/TIP3P.mol2')
        tip3p_mb.name = 'WAT'
        cation_mb = mbuild.load(f'{names.PROJECT_DIR}/files/coordinates/metal_cations/{job.sp.metal}.mol2')
        cation_mb.name = job.sp.metal
        cl_mb = mbuild.load(f'{names.PROJECT_DIR}/files/coordinates/neutralizing_anions/Cl.mol2')
        cl_mb.name = 'Cl'

        starting_box = mbuild.fill_box(
            compound=[tip3p_mb, cation_mb, cl_mb],
            n_compounds=[1000, 1, counterion_count],
            box=[3.8, 3.8, 3.8]
        )

        water_mol = Molecule.from_smiles('O')
        cl_mol = Molecule.from_smiles('[Cl-]')
        cation_rd = Chem.MolFromMol2File(
            f'{names.PROJECT_DIR}/files/coordinates/metal_cations/{job.sp.metal}.mol2', removeHs=False
        )
        cation_mol = Molecule.from_rdkit(cation_rd)
        cation_mol.atoms[0].formal_charge = names.METAL_FORMAL_CHARGES[job.sp.metal] * unit.elementary_charge

        mols = [water_mol] * 1000 + [cation_mol] + [cl_mol] * counterion_count
        topology = Topology.from_molecules(mols)
        topology.box_vectors = np.eye(3) * 3.8 * unit.nanometer

        ff = ForceField('tip3p.offxml', f'{names.PROJECT_DIR}/files/xml/custom_ree.offxml')
        interchange = ff.create_interchange(topology)
        interchange.positions = starting_box.xyz * unit.nanometer
        interchange.to_gromacs(prefix='init')

        if os.path.exists('init_pointenergy.mdp'):
            os.remove('init_pointenergy.mdp')

    current_lambda = names.eleLam_ljLam_to_initLam[round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]

    parameters = {
        "integrator": "sd",
        "dt": 0.001,
        "nsteps": names.MID_EQ_STEPS,
        "output_control": names.SLOW_OUTPUT,
        "nstcalcenergy": names.NORMAL_CALC,
        "rcoulomb": names.RCUT,
        "coulombtype": "PME",
        "coulomb_modifier": "None",
        "rcoulomb_switch": 0.0,
        "vdwtype": "Cut-off",
        "vdw_modifier": "None",
        "rvdw": names.RCUT,
        "rvdw_switch": 0.0,
        "DispCorr": "EnerPres",
        "tcoupl": "no",
        "ref_t": names.TEMPERATURE
    }

    misc_funct.simple_mdp_writer(
        job,
        mdp_name=f'{names.NAME_EQ_NVT}.mdp',
        parameters=parameters,
        constraints=None,
        templates_dir=f'{names.PROJECT_DIR}/files/mdp/',
        template_name='NVT_template.mdp'
    )

    parameters.update({
        "nsteps": names.LONG_EQ_STEPS,
        "output_control": names.SLOW_OUTPUT,
        "nstcalcenergy": names.NORMAL_CALC,
        "ref_t": names.TEMPERATURE,
        "pcoupl": "Berendsen",
        "ref_p": names.PRESSURE,
        "compressibility": 4.5e-5
    })

    misc_funct.simple_mdp_writer(
        job,
        mdp_name=f'{names.NAME_EQ_NPT_BERENDSEN}.mdp',
        parameters=parameters,
        constraints=None,
        templates_dir=f'{names.PROJECT_DIR}/files/mdp/',
        template_name='NPTmdp_template.mdp'
    )

    parameters.update({
        "nsteps": names.SMALL_EQ_STEPS,
        "output_control": names.SLOW_OUTPUT,
        "nstcalcenergy": names.NORMAL_CALC,
        "current_lambda": current_lambda,
        "molecule_of_interest": job.sp.metal,
        "nstdhdl": int(names.NORMAL_CALC * 10)
    })

    misc_funct.simple_mdp_writer(
        job,
        mdp_name=f'{names.NAME_EQ_CANON}.mdp',
        parameters=parameters,
        constraints=None,
        templates_dir=f'{names.PROJECT_DIR}/files/mdp/',
        template_name='free_energy_NPTmdp_template.mdp'
    )

    parameters.update({
        "nsteps": names.PRO_FREE_ENERGY_STEPS,
        "output_control": names.FAST_OUTPUT,
        "nstcalcenergy": names.NORMAL_CALC,
        "current_lambda": current_lambda,
        "molecule_of_interest": job.sp.metal,
        "nstdhdl": int(names.NORMAL_CALC * 10)
    })

    misc_funct.simple_mdp_writer(
        job,
        mdp_name=f'{names.NAME_PRO_CANON}.mdp',
        parameters=parameters,
        constraints=None,
        templates_dir=f'{names.PROJECT_DIR}/files/mdp/',
        template_name='free_energy_NPTmdp_template.mdp'
    )


@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.post(eq_nvt_post)
@FlowProject.operation(directives={"np": int(SIM_CORES), "ngpu": 1, "memory": 3.2, "walltime": MID_HOURS}, with_job=True, cmd=True)
def EQ_NVT(job):
    build_mdp = str(f'{GROMACS_PREFIX} grompp -f {names.NAME_EQ_NVT}.mdp -c init.gro -p init.top -o {names.NAME_EQ_NVT}.tpr -maxwarn 999')
    run_gmx = str(f'{GROMACS_PREFIX} mdrun -nt {SIM_CORES} -deffnm {names.NAME_EQ_NVT}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    return run_command


@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(eq_nvt_post)
@FlowProject.post(eq_npt_post_beren)
@FlowProject.operation(directives={"np": int(SIM_CORES), "ngpu": 1, "memory": 3.2, "walltime": TWO_DAYS}, with_job=True, cmd=True)
def EQ_NPT_BERENDSEN(job):
    build_mdp = str(f'{GROMACS_PREFIX} grompp -f {names.NAME_EQ_NPT_BERENDSEN}.mdp -c {names.NAME_EQ_NVT}.gro -p init.top -o {names.NAME_EQ_NPT_BERENDSEN}.tpr -maxwarn 999')
    run_gmx = str(f'{GROMACS_PREFIX} mdrun -nt {SIM_CORES} -deffnm {names.NAME_EQ_NPT_BERENDSEN}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    return run_command


@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(eq_npt_post_beren)
@FlowProject.post(eq_canon_post)
@FlowProject.operation(directives={"np": int(SIM_CORES), "ngpu": 1, "memory": 3.2, "walltime": TWO_DAYS}, with_job=True, cmd=True)
def EQ_CANON(job):
    build_mdp = str(f'{GROMACS_PREFIX} grompp -f {names.NAME_EQ_CANON}.mdp -c {names.NAME_EQ_NPT_BERENDSEN}.gro -p init.top -o {names.NAME_EQ_CANON}.tpr -maxwarn 999')
    run_gmx = str(f'{GROMACS_PREFIX} mdrun -nt {SIM_CORES} -deffnm {names.NAME_EQ_CANON}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    return run_command


@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(eq_canon_post)
@FlowProject.post(pro_canon_post)
@FlowProject.operation(directives={"np": int(SIM_CORES), "ngpu": 1, "memory": 3.2, "walltime": TWO_DAYS}, with_job=True, cmd=True)
def PRO_CANON(job):
    build_mdp = str(f'{GROMACS_PREFIX} grompp -f {names.NAME_PRO_CANON}.mdp -c {names.NAME_EQ_CANON}.gro -p init.top -o {names.NAME_PRO_CANON}.tpr -maxwarn 999')
    run_gmx = str(f'{GROMACS_PREFIX} mdrun -nt {SIM_CORES} -deffnm {names.NAME_PRO_CANON}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    return run_command


@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(pro_canon_post)
@FlowProject.post(free_energy_bar_copied)
@FlowProject.operation(directives={"np": int(SIM_CORES), "ngpu": 0, "memory": 3.2, "walltime": MIN_HOURS}, with_job=True, cmd=True)
def FREE_ENERGY_FILES_RENAMED(job):
    current_lambda = names.eleLam_ljLam_to_initLam[round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]
    run_command = str(f'cp {names.NAME_PRO_CANON}.xvg {names.NAME_PRO_CANON}_{current_lambda}.xvg')
    return run_command


@FlowProject.pre(data_collected)  # DUMMY TO AVOID THE JOB
@FlowProject.pre(free_energy_bar_copied)
@FlowProject.pre(pro_canon_post)
@FlowProject.post(data_collected)
@FlowProject.operation(directives={"np": int(ANA_CORES), "ngpu": 0, "memory": 1.1, "walltime": MIN_HOURS})
def GRAPH_AND_COLLECT_PROPERTIES(job):
    with job:
        properties_of_interest = ["Potential", "Pressure", "Total-Energy", "Temperature", "Density"]
        properties_of_interest_to_search_string_dict = {
            properties_of_interest[0]: ['Potential', '(kJ/mol)'],
            properties_of_interest[1]: ['Pressure', '(bar)'],
            properties_of_interest[2]: ['Total Energy', '(kJ/mol)'],
            properties_of_interest[3]: ['Temperature', '(K)'],
            properties_of_interest[4]: ['Density', '(kg/m^3)']
        }

        properties_of_interest_storage_dict = {
            properties_of_interest[0]: 0.0,
            properties_of_interest[1]: 0.0,
            properties_of_interest[2]: 0.0,
            properties_of_interest[3]: 0.0,
            properties_of_interest[4]: 0.0
        }

        gromacs_input = b'1\n0\n'
        result = subprocess.run(
            [f"{GROMACS_PREFIX}", "energy", "-f", f"{names.NAME_PRO_CANON}.edr", "-o", "dummy_data.xvg"],
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

        p = subprocess.Popen(
            [f'{GROMACS_PREFIX}', '-quiet', 'energy', '-f', f'{names.NAME_PRO_CANON}.edr', '-o', f'{names.GENERAL_LOCAL_DATA}_{names.NAME_PRO_CANON}.xvg'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        out, err = p.communicate(f'{newline_string}'.encode('utf-8'))
        capture = out.decode()

        with open(f'{names.GENERAL_LOCAL_DATA}_{names.NAME_PRO_CANON}.txt', 'w') as Dummy_GMX_output:
            Dummy_GMX_output.write(capture)

        with open(f'{names.GENERAL_LOCAL_DATA}_{names.NAME_PRO_CANON}.txt', 'r') as Dummy_GMX_output:
            with open(f"../../{names.GENERAL_GLOBAL_DATA}.txt", 'a') as aggregate_surTenFile:
                for a_single_line in Dummy_GMX_output:
                    for property_str in properties_of_interest:
                        search_property_str_dict = properties_of_interest_to_search_string_dict[property_str]
                        search_str_start = search_property_str_dict[0]
                        search_str_end = search_property_str_dict[1]

                        if (search_str_start in a_single_line) and (search_str_end in a_single_line):
                            numpyCatcher = np.fromstring(
                                a_single_line.strip(f'{search_str_start}{search_str_end}'),
                                dtype=float,
                                sep=' '
                            )[0]
                            properties_of_interest_storage_dict[property_str] = numpyCatcher

                aggregate_surTenFile.write(
                    f"{job.id:<42} {job.sp.lambda_LJ:<8} {job.sp.lambda_ELE:<8} {properties_of_interest_storage_dict['Potential']:<42} "
                    f" {properties_of_interest_storage_dict['Pressure']:<42} "
                    f" {properties_of_interest_storage_dict['Total-Energy']:<42} "
                    f" {properties_of_interest_storage_dict['Temperature']:<42} "
                    f" {properties_of_interest_storage_dict['Density']:<42} \n"
                )

        with open(f'{names.GENERAL_LOCAL_DATA}_{names.NAME_PRO_CANON}.xvg', 'r') as xvg_png_datasource:
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
        plt.savefig(f'{names.GENERAL_LOCAL_DATA}_{names.NAME_PRO_CANON}.png')
        plt.close()


if __name__ == '__main__':
    FlowProject().main()
