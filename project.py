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

from files.python_files import names, misc_funct
from files.python_files.misc_funct import (
    simple_mdp_writer,
    calculate_free_energy,
    calculate_rdf_freud,
    calculate_distance_time
)
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
    aggregated_data_present,
    distance_time_calculated,
    distance_data_written,
)

# Cores configuration
BUILD_CORES = int(1)
SIM_CORES = int(4)
ANA_CORES = int(2)

# Walltimes configuration
MIN_HOURS = 2.0
MID_HOURS = 8.0
DAY_WAIT = 24.0
TWO_DAYS = 48.0
ONE_WEEK = 168.0
TWO_WEEKS = 336.0

SIM_MEM = 2.5
BLD_MEM = 1.5
ANA_MEM = 2.0

project = signac.get_project()


class Custom_environment(DefaultSlurmEnvironment):
    hostname_pattern = r".*\.grid\.wayne\.edu"
    template = 'gmx_grid_fall2025.sh' # "gmx_grid_fall2025.sh"


@FlowProject.post(init_written)
@FlowProject.post(mdp_written)
@FlowProject.operation(directives={"np": BUILD_CORES, "ngpu": 0, "memory": BLD_MEM, "walltime": MIN_HOURS})
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
        box_length_multiplier = 1.0 #1.83
        box_length = init_box_length * box_length_multiplier
        n_water_molecules = int(init_n_water_molecules * (box_length_multiplier ** 3))
        

        # Load small molecules via RDKit
        water_rd = Chem.MolFromMol2File(
            f'{names.PROJECT_DIR}/files/coordinates/TIP4P_EW_No_M.mol2', removeHs=False
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
            #'ff14sb_off_impropers_0.0.4.offxml',   # protein residue typing + library charges
            f'{names.PROJECT_DIR}/files/xml/polypep/ff14sb_off_impropers_0.0.4.offxml',
            f'{names.PROJECT_DIR}/files/xml/water/tip4p_ew.offxml',                          # water
            f'{names.PROJECT_DIR}/files/xml/ions/custom_ree_tip4p_ew.offxml'  # REE ions (including TB in peptide)
        )

        interchange = ff.create_interchange(topology)
        interchange.to_gromacs(prefix='init')

        if os.path.exists('init_pointenergy.mdp'):
            os.remove('init_pointenergy.mdp')

        if job.sp.unNested_usesTemplates:
            try:
                import MDAnalysis as mda
                u = mda.Universe(f"{names.PROJECT_DIR}/files/coordinates/equilibrated_frames/{names.NAME_PRE_EQ_NPT_BERENDSEN}.gro")
                nd = u.select_atoms("name Tb")
                nd.names = [job.sp.metal]
                nd.residues.resnames = [job.sp.metal]
                u.atoms.write(f"{names.NAME_PRE_EQ_NPT_BERENDSEN}.gro")
            except:
                # Fallback: simple string replacement (no MDAnalysis needed)
                gro_file = f"{names.PROJECT_DIR}/files/coordinates/equilibrated_frames/{names.NAME_PRE_EQ_NPT_BERENDSEN}.gro"
                with open(gro_file, 'r') as f:
                    content = f.read()
                gro_file = f"{names.NAME_PRE_EQ_NPT_BERENDSEN}.gro"
                content = content.replace("Tb", job.sp.metal)
                with open(gro_file, 'w') as f:
                    f.write(content)

        if 'LBT3-' in job.sp.polypeptide:
            load_addendum = f'{names.PROJECT_DIR}/files/addendums/appendFEP_chelate_init_LBT3-.top'
        elif 'LBT5-' in job.sp.polypeptide:
            load_addendum = f'{names.PROJECT_DIR}/files/addendums/appendFEP_chelate_init_LBT5-.top'


        if 'LBT3-' in job.sp.polypeptide or 'LBT5-' in job.sp.polypeptide:
            # Backup init.top as original_init.top
            shutil.copy2('init.top', 'original_init.top')
            # Append the addendum to init.top
            with open(load_addendum, 'r') as addendum_file:
                addendum_content = addendum_file.read()
            with open('init.top', 'a') as init_top_file:
                init_top_file.write('\n')
                init_top_file.write(addendum_content)


        CHARGE_MULTIPLIER = 0.9

        if 'LBT5-' in job.sp.polypeptide and CHARGE_MULTIPLIER != 1.0:
            pairs_path = f'{names.PROJECT_DIR}/files/addendums/charges/pairs_LBT5-.txt'
            o_charges_path = f'{names.PROJECT_DIR}/files/addendums/charges/custom_charges_LBT5-_Oxygen.top'
            c_charges_path = f'{names.PROJECT_DIR}/files/addendums/charges/custom_charges_LBT5-_Carbon.top'

            with open(pairs_path, 'r') as f:
                pairs_lines = f.readlines()
            
            with open(o_charges_path, 'r') as f:
                o_lines = f.readlines()
                
            with open(c_charges_path, 'r') as f:
                c_lines = f.readlines()
            
            o_data = {}
            for line in o_lines:
                if not line.strip(): continue
                parts = line.split()
                atom_id = parts[0]
                charge = float(line.split(';')[1].split()[0])
                o_data[atom_id] = {'line': line, 'charge': charge}

            c_data = {}
            for line in c_lines:
                if not line.strip(): continue
                parts = line.split()
                atom_id = parts[0]
                charge = float(line.split(';')[1].split()[0])
                c_data[atom_id] = {'line': line, 'charge': charge}

            pair_groups = {}
            for line in pairs_lines:
                if not line.strip(): continue
                c_id, o_id = line.split()
                if c_id not in pair_groups:
                    pair_groups[c_id] = []
                pair_groups[c_id].append(o_id)
            
            new_lines_by_prefix = {}
            for c_id, o_ids in pair_groups.items():
                if c_id not in c_data:
                    continue
                c_charge_in = c_data[c_id]['charge']
                
                valid_o_ids = [oid for oid in o_ids if oid in o_data]
                if not valid_o_ids:
                    continue
                
                o_charges_in = [o_data[oid]['charge'] for oid in valid_o_ids]
                o_charges_out = [chg * CHARGE_MULTIPLIER for chg in o_charges_in]
                
                allC_charges_in = c_charge_in
                all_individual_O_charge_in = sum(o_charges_in)
                
                all_individual_O_charge_out = sum(o_charges_out)
                total_o_change = all_individual_O_charge_in - all_individual_O_charge_out
                allC_charges_out = c_charge_in + total_o_change
                
                all_pairs_in_pairGroup = f"[{c_id}, {', '.join(valid_o_ids)}]"
                
                print(f'{all_pairs_in_pairGroup} {allC_charges_in}+{all_individual_O_charge_in}={allC_charges_in+all_individual_O_charge_in}={allC_charges_out}+{all_individual_O_charge_out}={allC_charges_out+all_individual_O_charge_out}')
                
                for i, oid in enumerate(valid_o_ids):
                    o_line_template = o_data[oid]['line']
                    base_str = o_line_template.split(';')[0]
                    new_o_line = base_str.replace('{{charge}}', f"{o_charges_out[i]:.12f}").rstrip() + '\n'
                    prefix = o_line_template[:27]
                    new_lines_by_prefix[prefix] = new_o_line
                    
                c_line_template = c_data[c_id]['line']
                base_str = c_line_template.split(';')[0]
                new_c_line = base_str.replace('{{charge}}', f"{allC_charges_out:.12f}").rstrip() + '\n'
                prefix = c_line_template[:27]
                new_lines_by_prefix[prefix] = new_c_line
            
            with open('init.top', 'r') as f:
                init_top_lines = f.readlines()
                
            for i, line in enumerate(init_top_lines):
                prefix = line[:27]
                if prefix in new_lines_by_prefix:
                    init_top_lines[i] = new_lines_by_prefix[prefix]
                    
            with open('init.top', 'w') as f:
                f.writelines(init_top_lines)


    local_eleLam_ljLam_to_initLam = names.eleLam_ljLam_to_initLam
    current_lambda = local_eleLam_ljLam_to_initLam[round(job.sp.lambda_BONDED, 5), round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]
    sorted_lambda_states = sorted(
    local_eleLam_ljLam_to_initLam.items(),
    key=lambda x: x[1]
    )

    lambda_index = " ".join(f"{idx:>6}"      for (bonded_ele_lj, idx) in sorted_lambda_states)
    bonded_lambdas = " ".join(f"{bonded:>6.3f}" for (bonded, ele, lj), idx in sorted_lambda_states)
    coul_lambdas = " ".join(f"{ele:>6.3f}"   for (bonded, ele, lj), idx in sorted_lambda_states)
    vdw_lambdas  = " ".join(f"{lj:>6.3f}"    for (bonded, ele, lj), idx in sorted_lambda_states)
    
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
        "vdw_modifier": "Potential-shift", #"None",
        "rvdw": names.RCUT,
        "rvdw_switch": 0.0,
        "DispCorr": "EnerPres", #'no',
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
        "bonded_lambdas" : bonded_lambdas,
        "coul_lambdas" : coul_lambdas,
        "vdw_lambdas"  : vdw_lambdas,
    })

    misc_funct.simple_mdp_writer(
        job,
        mdp_name=f'{names.NAME_EQ_CANON}.mdp',
        parameters=parameters,
        constraints=None,
        templates_dir=f'{names.PROJECT_DIR}/files/mdp/',
        template_name='FEP_chelate_NVT_template.mdp'
    )

    parameters.update({
        "nsteps": names.PRO_FREE_ENERGY_STEPS,
        "output_control": names.FAST_OUTPUT,
        "nstcalcenergy": names.FAST_CALC,
        "current_lambda": current_lambda,
        "molecule_of_interest": job.sp.metal,
        "nstdhdl": int(names.PRO_DHDL),
        "lambda_index" : lambda_index,
        "bonded_lambdas" : bonded_lambdas,
        "coul_lambdas" : coul_lambdas,
        "vdw_lambdas"  : vdw_lambdas,
    })

    misc_funct.simple_mdp_writer(
        job,
        mdp_name=f'{names.NAME_PRO_CANON}.mdp',
        parameters=parameters,
        constraints=None,
        templates_dir=f'{names.PROJECT_DIR}/files/mdp/',
        template_name='FEP_chelate_NVT_template.mdp'
    )


@FlowProject.pre(runWithTemplateAbsent)
@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.post(templatedOrEquilibrated_eqNVT)
@FlowProject.operation(directives={"np": SIM_CORES, "ngpu": 1, "memory": SIM_MEM, "walltime": MID_HOURS}, with_job=True, cmd=True)
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
@FlowProject.operation(directives={"np": SIM_CORES, "ngpu": 1, "memory": SIM_MEM, "walltime": TWO_DAYS}, with_job=True, cmd=True)
def EQ_NPT_BERENDSEN(job):
    build_mdp = str(f'{names.GMX_PREFIX} grompp -f {names.NAME_EQ_NPT_BERENDSEN}.mdp -c {names.NAME_EQ_NVT}.gro -p init.top -o {names.NAME_EQ_NPT_BERENDSEN}.tpr -maxwarn 99')
    run_gmx = str(f'{names.GMX_PREFIX} mdrun -nt {SIM_CORES} -deffnm {names.NAME_EQ_NPT_BERENDSEN}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    return run_command


@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(templatedOrEquilibrated_eqNPT)
@FlowProject.post(eq_canon_post)
@FlowProject.operation(directives={"np": SIM_CORES, "ngpu": 1, "memory": SIM_MEM, "walltime": TWO_DAYS}, with_job=True, cmd=True)
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
@FlowProject.operation(directives={"np": SIM_CORES, "ngpu": 1, "memory": SIM_MEM, "walltime": TWO_DAYS}, with_job=True, cmd=True)
def PRO_CANON(job):
    build_mdp = str(f'{names.GMX_PREFIX} grompp -f {names.NAME_PRO_CANON}.mdp -c {names.NAME_EQ_CANON}.gro -p init.top -o {names.NAME_PRO_CANON}.tpr -maxwarn 99')
    run_gmx = str(f'{names.GMX_PREFIX} mdrun -nt {SIM_CORES} -deffnm {names.NAME_PRO_CANON}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    return run_command


@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(pro_canon_post)
@FlowProject.post(free_energy_bar_copied)
@FlowProject.operation(directives={"np": int(1), "ngpu": 0, "memory": SIM_MEM, "walltime": MIN_HOURS}, with_job=True, cmd=True)
def FREE_ENERGY_FILES_RENAMED(job):
    current_lambda = names.eleLam_ljLam_to_initLam[round(job.sp.lambda_BONDED, 5), round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]
    run_command = str(f'cp {names.NAME_PRO_CANON}.xvg {names.NAME_PRO_CANON}_{current_lambda}.xvg')
    return run_command


@FlowProject.pre(data_collected)  # DUMMY TO AVOID THE JOB
@FlowProject.pre(free_energy_bar_copied)
@FlowProject.pre(pro_canon_post)
@FlowProject.post(data_collected)
@FlowProject.operation(directives={"np": ANA_CORES, "ngpu": 0, "memory": ANA_MEM, "walltime": MIN_HOURS})
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
    directives={"np": ANA_CORES, "ngpu": 0, "memory": ANA_MEM, "walltime": MIN_HOURS},
    aggregator=aggregator.groupby(key=lambda job: (job.sp.metal, job.sp.polypeptide, job.sp.unNested_usesTemplates, job.sp.replicate))
)
def AGGREGATE_FREE_ENERGY(*jobs):
    group_parts = [str(jobs[0].sp.metal)]
    if getattr(jobs[0].sp, 'polypeptide', None):
        group_parts.append(str(jobs[0].sp.polypeptide))
    group_parts.append(str(jobs[0].sp.replicate))
    group_parts.append(str(jobs[0].sp.unNested_usesTemplates))
    group_name = "_".join(group_parts)
    group_dir = os.path.join(names.PROJECT_DIR, names.ANALYSIS_DIR_PREFIX, group_name)   # define group_dir
    target_dir = os.path.join(group_dir, "HFE")
    
    for job in jobs:
        current_lambda = names.eleLam_ljLam_to_initLam[round(job.sp.lambda_BONDED, 5), round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]

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

#DeepSeek RDF analysis V3
@FlowProject.pre(job_tester.pro_canon_post)
@FlowProject.post(job_tester.rdf_calculated)
@FlowProject.operation(
    directives={"np": ANA_CORES, "ngpu": 0, "memory": ANA_MEM, "walltime": MIN_HOURS}
)
def CALCULATE_RDF(job, debug_mode=True):
    group_parts = [str(job.sp.metal)]
    if getattr(job.sp, 'polypeptide', None):
        group_parts.append(str(job.sp.polypeptide))
    group_parts.append(str(job.sp.replicate))
    group_parts.append(str(job.sp.unNested_usesTemplates))
    group_name = "_".join(group_parts)
    target_dir = os.path.join(names.PROJECT_DIR, names.ANALYSIS_DIR_PREFIX, group_name)
    os.makedirs(target_dir, exist_ok=True)
    output_file = os.path.join(target_dir, "rdf_analysis.txt")
    
    result = misc_funct.calculate_rdf_freud(job, debug_mode=debug_mode, target_dir=target_dir)
    if result[0] is not None:
        min_1, max_1, cn_1, min_2, max_2, cn_2, min_3, max_3, cn_3 = result
        sp_str = f"{job.sp.metal}_{job.sp.polypeptide}_{job.sp.replicate} {job.sp.lambda_ELE:>6.4f} {job.sp.lambda_LJ:>6.4f}"
        with open(output_file, 'a') as f:
            # Write all three shells: min, max, CN per shell
            f.write(f"{job.id}, {min_1:.4f}, {max_1:.4f}, {cn_1:.4f}, "
                    f"{min_2:.4f}, {max_2:.4f}, {cn_2:.4f}, "
                    f"{min_3:.4f}, {max_3:.4f}, {cn_3:.4f}, {sp_str}\n")
        
        # ---- Summary file for lambda (0,0) with nice formatting ----
        if job.sp.lambda_ELE == 0.0 and job.sp.lambda_LJ == 0.0:
            summary_path = os.path.join(names.PROJECT_DIR, "lambda_0_0_rdf_summary.txt")
            
            # Column widths (increase as needed)
            col_widths = {
                'job_id': 34,
                'metal': 6,
                'min_1': 10,
                'max_1': 10,
                'cn_1': 10,
                'min_2': 10,
                'max_2': 10,
                'cn_2': 10,
                'min_3': 10,
                'max_3': 10,
                'cn_3': 10,
            }
            
            if not os.path.exists(summary_path):
                with open(summary_path, 'w') as sf:
                    header = (f"{'job_id':<{col_widths['job_id']}} "
                              f"{'metal':<{col_widths['metal']}} "
                              f"{'min_1':<{col_widths['min_1']}} "
                              f"{'max_1':<{col_widths['max_1']}} "
                              f"{'cn_1':<{col_widths['cn_1']}} "
                              f"{'min_2':<{col_widths['min_2']}} "
                              f"{'max_2':<{col_widths['max_2']}} "
                              f"{'cn_2':<{col_widths['cn_2']}} "
                              f"{'min_3':<{col_widths['min_3']}} "
                              f"{'max_3':<{col_widths['max_3']}} "
                              f"{'cn_3':<{col_widths['cn_3']}}")
                    sf.write(header + "\n")
            
            with open(summary_path, 'a') as sf:
                row = (f"{job.id:<{col_widths['job_id']}} "
                       f"{job.sp.metal:<{col_widths['metal']}} "
                       f"{min_1:<{col_widths['min_1']}.4f} "
                       f"{max_1:<{col_widths['max_1']}.4f} "
                       f"{cn_1:<{col_widths['cn_1']}.4f} "
                       f"{min_2:<{col_widths['min_2']}.4f} "
                       f"{max_2:<{col_widths['max_2']}.4f} "
                       f"{cn_2:<{col_widths['cn_2']}.4f} "
                       f"{min_3:<{col_widths['min_3']}.4f} "
                       f"{max_3:<{col_widths['max_3']}.4f} "
                       f"{cn_3:<{col_widths['cn_3']}.4f}")
                sf.write(row + "\n")

@FlowProject.pre(pro_canon_post)
@FlowProject.post(distance_data_written)
@FlowProject.operation(directives={"np": ANA_CORES, "ngpu": 0, "memory": ANA_MEM, "walltime": MIN_HOURS})
def CALCULATE_DISTANCE_TIME(job):
    group_parts = [str(job.sp.metal)]
    if getattr(job.sp, 'polypeptide', None):
        group_parts.append(str(job.sp.polypeptide))
    group_parts.append(str(job.sp.replicate))
    group_parts.append(str(job.sp.unNested_usesTemplates))
    group_name = "_".join(group_parts)
    group_dir = os.path.join(names.PROJECT_DIR, names.ANALYSIS_DIR_PREFIX, group_name)
    target_dir = os.path.join(group_dir, "distance")
    os.makedirs(target_dir, exist_ok=True)
    calculate_distance_time(job, target_dir)


@FlowProject.pre(distance_data_written)
@FlowProject.post(distance_time_calculated)
@FlowProject.operation(
    directives={"np": ANA_CORES, "ngpu": 0, "memory": ANA_MEM, "walltime": MIN_HOURS},
    aggregator=aggregator.groupby(key=lambda job: (job.sp.metal, job.sp.polypeptide, job.sp.replicate, job.sp.unNested_usesTemplates))
)
def AGGREGATE_DISTANCE_TIME(*jobs):
    import glob
    import re
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    first_job = jobs[0]
    group_parts = [
        str(first_job.sp.metal),
        str(first_job.sp.polypeptide) if getattr(first_job.sp, 'polypeptide', None) else 'None',
        str(first_job.sp.replicate),
        str(first_job.sp.unNested_usesTemplates)
    ]
    group_name = "_".join(group_parts)
    distance_dir = os.path.join(names.PROJECT_DIR, names.ANALYSIS_DIR_PREFIX, group_name, "distance")
    os.makedirs(distance_dir, exist_ok=True)

    print(f"Processing group: {group_name}")

    def read_distance_file(fpath):
        try:
            df = pd.read_csv(fpath, sep=r'\s+', header=None, skiprows=1,
                             names=['time', 'dist'])
            if len(df) < 2:
                return None, None
            time = df['time'].astype(float).values
            dist = df['dist'].astype(float).values
            return time, dist
        except Exception as e:
            print(f"  Error reading {fpath}: {e}")
            return None, None

    # Collect per‑atom data (CA and O_coord)
    atom_files = glob.glob(os.path.join(distance_dir, "distance_*_serial*_lam*.txt"))
    print(f"Found {len(atom_files)} per‑atom data files")
    atom_data = {}

    for fpath in atom_files:
        base = os.path.basename(fpath)
        parts = base.split('_')
        # label is everything between 'distance' and 'serial'
        label = '_'.join(parts[1:-2])   # e.g., 'CA' or 'O_coord'
        serial = None
        lambda_idx = None
        for part in parts:
            if part.startswith('serial'):
                serial = int(part.replace('serial', ''))
            elif part.startswith('lam'):
                m = re.search(r'lam(\d+)', part)
                if m:
                    lambda_idx = int(m.group(1))
        if label is None or serial is None or lambda_idx is None:
            print(f"  Skipping file {base}: could not parse label/serial/lambda")
            continue
        time, dist = read_distance_file(fpath)
        if time is None:
            continue
        atom_data.setdefault((label, serial), []).append((lambda_idx, time, dist))
        print(f"  Added {label} serial {serial} lambda {lambda_idx}")

    print(f"Collected {len(atom_data)} atom types/serials with data")

    # Collect OHN mean data
    ohn_files = glob.glob(os.path.join(distance_dir, "distance_OHN_lam*.txt"))
    print(f"Found {len(ohn_files)} OHN data files")
    ohn_data = []
    for fpath in ohn_files:
        base = os.path.basename(fpath)
        parts = base.split('_')
        lambda_idx = None
        for part in parts:
            if part.startswith('lam'):
                m = re.search(r'lam(\d+)', part)
                if m:
                    lambda_idx = int(m.group(1))
        if lambda_idx is None:
            continue
        time, dist = read_distance_file(fpath)
        if time is None:
            continue
        ohn_data.append((lambda_idx, time, dist))
        print(f"  Added OHN lambda {lambda_idx}")

    print(f"Collected {len(ohn_data)} OHN lambdas with data")

    if not atom_data and not ohn_data:
        print(f"WARNING: No data found for group {group_name}. No plots created.")
        return

    plots_created = False
    if atom_data:
        cmap = plt.get_cmap('plasma')
        for (label, serial), data_list in atom_data.items():
            data_list.sort(key=lambda x: x[0])
            lambdas = [d[0] for d in data_list]
            norm = Normalize(vmin=min(lambdas), vmax=max(lambdas))
            fig, ax = plt.subplots(figsize=(10, 6))
            for lambda_idx, time, dist in data_list:
                ax.plot(time, dist, color=cmap(norm(lambda_idx)), linewidth=1.5, alpha=0.8)

            first_lambda_str = f"lam{data_list[0][0]:02d}"
            start_file = os.path.join(distance_dir, f"pro_canon_start_time_{first_lambda_str}.txt")
            if os.path.exists(start_file):
                with open(start_file, 'r') as f:
                    start_time = float(f.read().strip())
                ax.axvline(x=start_time, color='black', linestyle='--', linewidth=2, label='PRO_CANON start')

            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('λ-index')
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Distance (nm)')
            ax.set_title(f'{group_name} – {label} serial {serial}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            out_file = os.path.join(distance_dir, f"overlay_{label}_serial{serial}.png")
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Overlay saved: {out_file}")
            plots_created = True

    if ohn_data:
        ohn_data.sort(key=lambda x: x[0])
        lambdas = [d[0] for d in ohn_data]
        norm = Normalize(vmin=min(lambdas), vmax=max(lambdas))
        fig, ax = plt.subplots(figsize=(10, 6))
        for lambda_idx, time, dist in ohn_data:
            ax.plot(time, dist, color=cmap(norm(lambda_idx)), linewidth=1.5, alpha=0.8)

        first_lambda_str = f"lam{ohn_data[0][0]:02d}"
        start_file = os.path.join(distance_dir, f"pro_canon_start_time_{first_lambda_str}.txt")
        if os.path.exists(start_file):
            with open(start_file, 'r') as f:
                start_time = float(f.read().strip())
            ax.axvline(x=start_time, color='black', linestyle='--', linewidth=2, label='PRO_CANON start')

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('λ-index')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Mean Distance (nm)')
        ax.set_title(f'{group_name} – OHN (mean distance)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        out_file = os.path.join(distance_dir, "overlay_OHN.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Overlay saved: {out_file}")
        plots_created = True

    if plots_created:
        print("Cleaning up .txt files...")
        for f in glob.glob(os.path.join(distance_dir, "*.txt")):
            try:
                os.remove(f)
                print(f"  Removed {f}")
            except Exception as e:
                print(f"  Could not remove {f}: {e}")
        print("Cleanup complete.")
    else:
        print("No plots were created, keeping .txt files for debugging.")

if __name__ == '__main__':
    FlowProject().main()