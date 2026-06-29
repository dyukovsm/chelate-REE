"""
REE_HYDRATION signac workflow project.

Rare earth element hydration free energy simulations using GROMACS
with signac workflow management.

Contributors:
  - dyukovsm <go0719@wayne.edu> (Lead Developer)
  - Gemini (Google DeepMind) (Co-Author)
  - Claude Opus 4.5 (Anthropic) (Co-Author)
"""

from files.python_files import job_tester

import math
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import signac
# pyrefly: ignore [missing-import]
from flow import FlowProject, aggregator
# pyrefly: ignore [missing-import]
from flow.environment import DefaultSlurmEnvironment
import os
import shutil
import subprocess
import re
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import io

from files.python_files import names
from files.python_files import misc_funct
from files.python_files.job_tester import (
    init_written,
    mdp_written,
    eq_nvt_post,
    eq_npt_post_beren,
    runWithTemplateAbsent,
    templatedOrEquilibrated_eqNVT,
    templatedOrEquilibrated_eqNPT,
    eq_canon_post,
    pro_canon_post,
    free_energy_bar_copied,
    data_collected,
    xvg_present_for_all,
    aggregated_data_present
)

# Cores configuration
BUILD_CORES = 1
SIM_CORES = 4
ANA_CORES = 1

# Walltimes configuration
MIN_HOURS = 2.0
MID_HOURS = 8.0
DAY_WAIT = 24.0
TWO_DAYS = 48.0
ONE_WEEK = 168.0
TWO_WEEKS = 336.0


project = signac.get_project()


class Custom_environment(DefaultSlurmEnvironment):
    hostname_pattern = r".*\.grid\.wayne\.edu"
    template = "gmx_grid_fall2025.sh"


@FlowProject.post(init_written)
@FlowProject.post(mdp_written)
@FlowProject.operation(directives={"np": int(BUILD_CORES), "ngpu": 0, "memory": 3.2, "walltime": MIN_HOURS})
def build_input(job):
    with job:
        import sys
        # pyrefly: ignore [missing-import]
        from rdkit import Chem
        # pyrefly: ignore [missing-import]
        from openff.toolkit import Molecule, Topology, ForceField
        # pyrefly: ignore [missing-import]
        from openff.units import unit
        # pyrefly: ignore [missing-import]
        from openff.interchange.components._packmol import pack_box

        conda_bin = os.path.dirname(sys.executable)
        os.environ['PATH'] = f'{conda_bin}:{os.environ.get("PATH", "")}'
        
        
        if job.sp.metal in ('U', 'Hf'):
            metal_ion_charge = 4
        elif job.sp.metal in ('Ca', 'Mg', 'Zn', 'Cu', 'Ni'):
            metal_ion_charge = 2
        else:
            metal_ion_charge = 3

        if job.sp.polypeptide == 'LBT3-':
            polypeptide_charge = -3
        elif job.sp.polypeptide == 'LBT5-':
            polypeptide_charge = -5
        elif job.sp.polypeptide == 'DUM3+':
            polypeptide_charge = 0

        net_ion_charge = metal_ion_charge + polypeptide_charge; net_ion_charge=int(net_ion_charge)
        print(f"Metal: {job.sp.metal}, Polypeptide: {job.sp.polypeptide}, Net Ion Charge: {net_ion_charge}")
        if net_ion_charge > 0:
            counterion_str = 'Cl'
        elif net_ion_charge < 0:
            counterion_str = 'Li'
        counterion_count = abs(net_ion_charge)

        init_box_length = 3.5  # nm
        init_n_water_molecules = 1000
        box_length_multiplier = 1.83
        box_length = init_box_length * box_length_multiplier
        n_water_molecules = int(init_n_water_molecules * (box_length_multiplier ** 3))
        

        # Load small molecules via RDKit
        water_rd = Chem.MolFromMol2File(
            f'{names.PROJECT_DIR}/files/coordinates/TIP3P.mol2', removeHs=False
        )
        water_mol = Molecule.from_rdkit(water_rd)

        if counterion_count != 0:
            li_rd = Chem.MolFromMol2File(
                f'{names.PROJECT_DIR}/files/coordinates/neutralizing_ions/{counterion_str}.mol2', removeHs=False
            )
            li_mol = Molecule.from_rdkit(li_rd)
            if 'Cl' in counterion_str:
                li_mol.atoms[0].formal_charge = -1 * unit.elementary_charge
            elif 'Li' in counterion_str:
                li_mol.atoms[0].formal_charge = 1 * unit.elementary_charge

        cation_rd = Chem.MolFromMol2File(
            f'{names.PROJECT_DIR}/files/coordinates/metal_cations/{job.sp.metal}.mol2', removeHs=False
        )
        cation_mol = Molecule.from_rdkit(cation_rd)
        cation_mol.atoms[0].formal_charge = names.METAL_FORMAL_CHARGES[job.sp.metal] * unit.elementary_charge

        # Handle dummy (pure ion + water) vs polypeptide cases
        if job.sp.polypeptide == 'DUM3+':
            # Dummy case: no polypeptide, just ion in water (no solute for pack_box)
            solute_topology = None
        else:
            # Extract TB ion coordinates from original PDB (TB is removed from cleaned files)
            original_pdb_path = f'{names.PROJECT_DIR}/files/coordinates/polypeptide/{job.sp.polypeptide}.pdb'
            tb_coords = None
            with open(original_pdb_path, 'r') as f:
                for line in f:
                    if line.startswith(('ATOM', 'HETATM')):
                        atom_name = line[12:16].strip()
                        res_name = line[17:20].strip()
                        if res_name == 'TB' or atom_name == 'TB':
                            tb_coords = (
                                float(line[30:38]),
                                float(line[38:46]),
                                float(line[46:54])
                            )
                            break

            # Create the ACTUAL target metal ion (not Tb) at the TB template position
            # This avoids having two metal ions (template Tb + target metal)
            metal_smiles = f'[{job.sp.metal}+{metal_ion_charge}]'
            target_metal_mol = Molecule.from_smiles(metal_smiles)
            target_metal_mol.atoms[0].formal_charge = metal_ion_charge * unit.elementary_charge
            if tb_coords is not None:
                target_metal_mol.add_conformer(np.array([tb_coords]) * unit.angstrom)

            # Load pre-cleaned polypeptide topology (cleaned by unscrew_polypeptide.py)
            cleaned_pdb_path = f'{names.PROJECT_DIR}/files/coordinates/polypeptide/{job.sp.polypeptide}{names.CLEANED_PDB_SUFFIX}.pdb'
            polypeptide_topology = Topology.from_pdb(cleaned_pdb_path)

            # Add target metal ion to the topology
            metal_topology = Topology.from_molecules([target_metal_mol])
            for mol in metal_topology.molecules:
                polypeptide_topology.add_molecule(mol)

            # Center the solute in the box to avoid coordinates outside box bounds
            # The original PDB coordinates are in crystallographic frame, need to translate
            box_center = np.array([box_length / 2, box_length / 2, box_length / 2])  # nm

            # Collect all atom positions to find centroid
            all_positions = []
            for mol in polypeptide_topology.molecules:
                if mol.n_conformers > 0:
                    # Convert to nm for consistency
                    positions_nm = mol.conformers[0].to(unit.nanometer).magnitude
                    all_positions.extend(positions_nm)

            if all_positions:
                all_positions = np.array(all_positions)
                centroid = np.mean(all_positions, axis=0)
                translation = box_center - centroid

                # Apply translation to all molecules in the topology
                for mol in polypeptide_topology.molecules:
                    if mol.n_conformers > 0:
                        old_conf = mol.conformers[0].to(unit.nanometer).magnitude
                        new_conf = old_conf + translation
                        # Clear and set new conformer
                        mol._conformers = []
                        mol.add_conformer(new_conf * unit.nanometer)

            solute_topology = polypeptide_topology

        # Build molecule list for pack_box
        # For polypeptide cases: metal is already in solute_topology, don't add separately
        # For DUM3+ case: metal must be added to molecules_dummy
        if solute_topology is not None:
            # Polypeptide case: metal already in solute, only add water + counterions
            if counterion_count != 0:
                molecules_dummy = [water_mol, li_mol]
                number_of_copies_dummy = [n_water_molecules, counterion_count]
            else:
                molecules_dummy = [water_mol]
                number_of_copies_dummy = [n_water_molecules]
        else:
            # DUM3+ case: need to add cation_mol since there's no solute with metal
            if counterion_count != 0:
                molecules_dummy = [water_mol, cation_mol, li_mol]
                number_of_copies_dummy = [n_water_molecules, 1, counterion_count]
            else:
                molecules_dummy = [water_mol, cation_mol]
                number_of_copies_dummy = [n_water_molecules, 1]

        # Pack the box using OpenFF's pack_box
        topology = pack_box(
            molecules=molecules_dummy,
            number_of_copies=number_of_copies_dummy,
            solute=solute_topology,
            box_vectors=np.eye(3) * box_length * unit.nanometer,
        )

        ff = ForceField(
            'ff14sb_off_impropers_0.0.4.offxml',   # protein residue typing + library charges
            'tip3p.offxml',                          # water
            f'{names.PROJECT_DIR}/files/xml/custom_ree.offxml'  # REE ions (including TB in peptide)
        )

        interchange = ff.create_interchange(topology)
        interchange.to_gromacs(prefix='init')

        if os.path.exists('init_pointenergy.mdp'):
            os.remove('init_pointenergy.mdp')

        if job.sp.unNested_usesTemplates:
            try:
                import MDAnalysis as mda
                u = mda.Universe(f"{names.NAME_PRE_EQ_NPT_BERENDSEN}.gro")
                nd = u.select_atoms("name Tb")
                nd.names = [job.sp.metal]
                nd.residues.resnames = [job.sp.metal]
                u.atoms.write(f"{names.NAME_PRE_EQ_NPT_BERENDSEN}.gro")
            except:
                # Fallback: simple string replacement (no MDAnalysis needed)
                gro_file = f"{names.NAME_PRE_EQ_NPT_BERENDSEN}.gro"
                with open(gro_file, 'r') as f:
                    content = f.read()
                content = content.replace("Tb", job.sp.metal)
                with open(gro_file, 'w') as f:
                    f.write(content)

    local_eleLam_ljLam_to_initLam = names.eleLam_ljLam_to_initLam
    current_lambda = local_eleLam_ljLam_to_initLam[round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]
    sorted_lambda_states = sorted(
    local_eleLam_ljLam_to_initLam.items(),
    key=lambda x: x[1]
    )

    lambda_index = " ".join(f"{idx:>6}"      for (ele_lj, idx) in sorted_lambda_states)
    coul_lambdas = " ".join(f"{ele:>6.3f}"   for (ele, lj), idx in sorted_lambda_states)
    vdw_lambdas  = " ".join(f"{lj:>6.3f}"    for (ele, lj), idx in sorted_lambda_states)
    
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
        "nstdhdl": int(names.NORMAL_CALC * 10),
        "lambda_index" : lambda_index,
        "coul_lambdas" : coul_lambdas,
        "vdw_lambdas"  : vdw_lambdas,
    })

    misc_funct.simple_mdp_writer(
        job,
        mdp_name=f'{names.NAME_EQ_CANON}.mdp',
        parameters=parameters,
        constraints=None,
        templates_dir=f'{names.PROJECT_DIR}/files/mdp/',
        template_name='free_energy_Canonical_mdp_template.mdp'
    )

    parameters.update({
        "nsteps": names.PRO_FREE_ENERGY_STEPS,
        "output_control": names.FAST_OUTPUT,
        "nstcalcenergy": names.FAST_CALC,
        "current_lambda": current_lambda,
        "molecule_of_interest": job.sp.metal,
        "nstdhdl": int(names.PRO_DHDL),
        "lambda_index" : lambda_index,
        "coul_lambdas" : coul_lambdas,
        "vdw_lambdas"  : vdw_lambdas,
    })

    misc_funct.simple_mdp_writer(
        job,
        mdp_name=f'{names.NAME_PRO_CANON}.mdp',
        parameters=parameters,
        constraints=None,
        templates_dir=f'{names.PROJECT_DIR}/files/mdp/',
        template_name='free_energy_Canonical_mdp_template.mdp'
    )


@FlowProject.pre(runWithTemplateAbsent)
@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.post(templatedOrEquilibrated_eqNVT)
@FlowProject.operation(directives={"np": int(SIM_CORES), "ngpu": 1, "memory": 3.2, "walltime": MID_HOURS}, with_job=True, cmd=True)
def EQ_NVT(job):
    build_mdp = str(f'{names.GMX_PREFIX} grompp -f {names.NAME_EQ_NVT}.mdp -c init.gro -p init.top -o {names.NAME_EQ_NVT}.tpr -maxwarn 99')
    run_gmx = str(f'{names.GMX_PREFIX} mdrun -nt {SIM_CORES} -deffnm {names.NAME_EQ_NVT}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    return run_command

@FlowProject.pre(runWithTemplateAbsent)
@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(eq_nvt_post)
@FlowProject.post(templatedOrEquilibrated_eqNPT)
@FlowProject.operation(directives={"np": int(SIM_CORES), "ngpu": 1, "memory": 3.2, "walltime": TWO_DAYS}, with_job=True, cmd=True)
def EQ_NPT_BERENDSEN(job):
    build_mdp = str(f'{names.GMX_PREFIX} grompp -f {names.NAME_EQ_NPT_BERENDSEN}.mdp -c {names.NAME_EQ_NVT}.gro -p init.top -o {names.NAME_EQ_NPT_BERENDSEN}.tpr -maxwarn 99')
    run_gmx = str(f'{names.GMX_PREFIX} mdrun -nt {SIM_CORES} -deffnm {names.NAME_EQ_NPT_BERENDSEN}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    return run_command


@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(templatedOrEquilibrated_eqNPT)
@FlowProject.post(eq_canon_post)
@FlowProject.operation(directives={"np": int(SIM_CORES), "ngpu": 1, "memory": 3.2, "walltime": TWO_DAYS}, with_job=True, cmd=True)
def EQ_CANON(job):
    if not(job.sp.unNested_usesTemplates):
    	build_mdp = str(f'{names.GMX_PREFIX} grompp -f {names.NAME_EQ_CANON}.mdp -c {names.NAME_EQ_NPT_BERENDSEN}.gro -p init.top -o {names.NAME_EQ_CANON}.tpr -maxwarn 99')
    else:
    	build_mdp = str(f'{names.GMX_PREFIX} grompp -f {names.NAME_EQ_CANON}.mdp -c {names.NAME_PRE_EQ_NPT_BERENDSEN}.gro -p init.top -o {names.NAME_EQ_CANON}.tpr -maxwarn 99')
    run_gmx = str(f'{names.GMX_PREFIX} mdrun -nt {SIM_CORES} -deffnm {names.NAME_EQ_CANON}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    return run_command


@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(eq_canon_post)
@FlowProject.post(pro_canon_post)
@FlowProject.operation(directives={"np": int(SIM_CORES), "ngpu": 1, "memory": 3.2, "walltime": TWO_DAYS}, with_job=True, cmd=True)
def PRO_CANON(job):
    build_mdp = str(f'{names.GMX_PREFIX} grompp -f {names.NAME_PRO_CANON}.mdp -c {names.NAME_EQ_CANON}.gro -p init.top -o {names.NAME_PRO_CANON}.tpr -maxwarn 99')
    run_gmx = str(f'{names.GMX_PREFIX} mdrun -nt {SIM_CORES} -deffnm {names.NAME_PRO_CANON}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    return run_command


@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(pro_canon_post)
@FlowProject.post(free_energy_bar_copied)
@FlowProject.operation(directives={"np": int(1), "ngpu": 0, "memory": 3.2, "walltime": MIN_HOURS}, with_job=True, cmd=True)
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
            [f"{names.GMX_PREFIX}", "energy", "-f", f"{names.NAME_PRO_CANON}.edr", "-o", "dummy_data.xvg"],
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
            [f'{names.GMX_PREFIX}', '-quiet', 'energy', '-f', f'{names.NAME_PRO_CANON}.edr', '-o', f'{names.GENERAL_LOCAL_DATA}_{names.NAME_PRO_CANON}.xvg'],
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


@FlowProject.pre(xvg_present_for_all)
@FlowProject.post(aggregated_data_present)
@FlowProject.operation(
    directives={"np": int(ANA_CORES), "ngpu": 0, "memory": 3.2, "walltime": MIN_HOURS},
    aggregator=aggregator.groupby(key=lambda job: (job.sp.metal, job.sp.polypeptide, job.sp.unNested_usesTemplates, job.sp.replicate))
)
def AGGREGATE_FREE_ENERGY(*jobs):
    group_parts = [str(jobs[0].sp.metal)]
    if getattr(jobs[0].sp, 'polypeptide', None):
        group_parts.append(str(jobs[0].sp.polypeptide))
    group_parts.append(str(jobs[0].sp.replicate))
    group_parts.append(str(jobs[0].sp.unNested_usesTemplates))
    group_name = "_".join(group_parts)
    target_dir = os.path.join(names.PROJECT_DIR, names.ANALYSIS_DIR_PREFIX, group_name)
    os.makedirs(target_dir, exist_ok=True)
    
    for job in jobs:
        current_lambda = names.eleLam_ljLam_to_initLam[round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]
        
        xvg_src = job.fn(f'{names.NAME_PRO_CANON}_{current_lambda}.xvg')
        mdp_src = job.fn(f'{names.NAME_PRO_CANON}.mdp')
        
        if os.path.exists(xvg_src):
            subprocess.run(f"cp -v {xvg_src} {target_dir}/", shell=True)
        if os.path.exists(mdp_src):
            subprocess.run(f"cp -v {mdp_src} {target_dir}/{names.NAME_PRO_CANON}_{current_lambda}.mdp", shell=True)

    mbar_val, mbar_err = misc_funct.calculate_free_energy(target_dir)

    output_file = os.path.join(names.PROJECT_DIR, "aggregated_free_energy.txt")
    with open(output_file, 'a') as f:
        f.write(f"{group_name}: {mbar_val} +/- {mbar_err}\n")


if __name__ == '__main__':
    FlowProject().main()
