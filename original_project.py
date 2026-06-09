import math
import mbuild 
import numpy as np
import signac
from flow import FlowProject
from flow.environment import DefaultSlurmEnvironment
import os
import shutil
import forcefield_utilities
import gmso
from gmso.external.convert_mbuild import from_mbuild
from gmso.parameterization import apply
from gmso.formats.top import write_top
from gmso.formats.gro import write_gro
from jinja2 import Environment, FileSystemLoader
import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import io

PROJECT_DIR = os.path.abspath('.')
INIT_FILE_LIST = ['init.gro','init.top']
WAT_STANDARD_NAME = "WATER"
GROMACS_PREFIX = "/usr/local/gromacs/bin/gmx"
#names
NAME_EM = "em"
NAME_EQ_NVT = "EQ_NVT"
NAME_EQ_NPT_BERENDSEN = "EQ_NPT_BERENDSEN"
NAME_EQ_NPT_PARRINELLO = "EQ_NPT_PARRINELLO"
NAME_PRO_NPT_PARRINELLO = "PRO_NPT_PARRINELLO"
TEMPERATURE = 298.0
PRESSURE = 1.0

MDP_FILE_LIST = [f'{NAME_EQ_NVT}.mdp', f'{NAME_EQ_NPT_BERENDSEN}.mdp', f'{NAME_EQ_NPT_PARRINELLO}.mdp', f'{NAME_PRO_NPT_PARRINELLO}.mdp']

ljLam_eleLam_to_initLam = {
    (0.0,  0.000) : 0,
    (0.2,  0.000) : 1,
    (0.4,  0.000) : 2,
    (0.6,  0.000) : 3,
    (0.8,  0.000) : 4,
    (0.9,  0.000) : 5,
    (1.0,  0.000) : 6,
    (1.0,  0.050) : 7,
    (1.0,  0.100) : 8,
    (1.0,  0.150) : 9,
    (1.0,  0.200) : 10,
    (1.0,  0.250) : 11,
    (1.0,  0.300) : 12,
    (1.0,  0.350) : 13,
    (1.0,  0.400) : 14,
    (1.0,  0.425) : 15,
    (1.0,  0.450) : 16,
    (1.0,  0.475) : 17,
    (1.0,  0.500) : 18,
    (1.0,  0.550) : 19,
    (1.0,  0.600) : 20,
    (1.0,  0.650) : 21,
    (1.0,  0.700) : 22,
    (1.0,  0.750) : 23,
    (1.0,  0.800) : 24,
    (1.0,  0.850) : 25,
    (1.0,  0.900) : 26,
    (1.0,  0.950) : 27,
    (1.0,  1.000) : 28
}

MIN_CORES = 1
MID_CORES = 4

#### test
SMALL_EQ_STEPS      = int(100000) 
MID_EQ_STEPS        = int(1000000)      
LONG_EQ_STEPS       = int(3000000)      
SLOW_OUTPUT         = int(20000)    
NORMAL_CALC         = int(100)      
 
PRO_STEPS           = int(500000) 
PRO_FREE_ENERGY_STEPS = int(500000)
FAST_OUTPUT         = int(1000) 
FAST_CALC           = int(100) 

## actual run
##SMALL_EQ_STEPS      = int(500000) 
##MID_EQ_STEPS        = int(2000000)      
##LONG_EQ_STEPS       = int(3000000)      
##SLOW_OUTPUT         = int(20000)    
##NORMAL_CALC         = int(100)      
##
##PRO_STEPS           = int(2000000) 
##PRO_FREE_ENERGY_STEPS = int(1000000)
##FAST_OUTPUT         = int(5000) 
##FAST_CALC           = int(50) 

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
            
## @FlowProject.label
## def em_post(job):
##     with(job):
##         test_passed = False
##         if job.isfile(f"em.log"):
##             file_with_lines = open(f"em.log","r")
##             lines = file_with_lines.readlines()
##             file_with_lines.close()
##             for single_line in lines:
##                 if "Received the " in single_line:
##                     if " signal, stopping within" in single_line:
##                         test_passed = False
##                         break
##                 
##                 if "Finished mdrun on " in single_line:
##                     test_passed = True
##                     break
##     
##     return test_passed  


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
def eq_npt_post_parrin(job):
    with(job):
        test_passed = False
        if job.isfile(f"{NAME_EQ_NPT_PARRINELLO}.log"):
            file_with_lines = open(f"{NAME_EQ_NPT_PARRINELLO}.log","r")
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
def pro_npt_post_parrin(job):
    with(job):
        test_passed = False
        if job.isfile(f"{NAME_PRO_NPT_PARRINELLO}.log"):
            file_with_lines = open(f"{NAME_PRO_NPT_PARRINELLO}.log","r")
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
        #file_of_interest = NAME_EQ_NPT_PARRINELLO
        test_passed = False
        if job.isfile(f'{NAME_PRO_NPT_PARRINELLO}.xvg'):
            current_lambda = ljLam_eleLam_to_initLam[round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]
            if job.isfile(f'{NAME_PRO_NPT_PARRINELLO}_{current_lambda}.xvg'):
                #pass
                test_passed = True
                #print(f'test passed inside free_energy_bar_copied')
 
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

    #hostname_pattern = r".*\.grid\.wayne\.edu"
    #template = "gmx_grid.sh"
    #template = "gmx_grid_with_secondary.sh"
    #template = "potoff_gpu_update.sh"
    #template = "potoff_prospectus.sh"
    template = "v3_2025_gpu_potoff.sh"

@FlowProject.post(init_written)
@FlowProject.post(mdp_written)
@FlowProject.operation(directives={ "np": int(1),  "ngpu": 0, "memory": 3.2, "walltime": MIN_HOURS})
def build_input(job):
    with(job):
        spce_coordinates = mbuild.load(f"{PROJECT_DIR}/files/coordinates/SPCE.mol2")
        spce_coordinates.name = "WAT"
        
        cation_coordinates = mbuild.load(f"{PROJECT_DIR}/files/coordinates/metal_cations/{job.sp.metal}.mol2")
        cation_coordinates.name = job.sp.metal
        
        counterion_coordinates = mbuild.load(f"{PROJECT_DIR}/files/coordinates/neutralizing_anions/Cl.mol2")
        counterion_coordinates.name = 'Cl'
        #spce_coordinates.save("init.gro")
        
        if job.sp.metal == 'U' or job.sp.metal == 'Hf':
            counterion_count = 4
        else:
            counterion_count = 3

        starting_box = mbuild.fill_box(compound=[spce_coordinates,cation_coordinates,counterion_coordinates],n_compounds=[1000,1,counterion_count],box=[3.8000,3.8000,3.8000]) # volume matched to mohammad's box --> box=[3.6700,3.6700,3.6700])
        wat_ff_xml = forcefield_utilities.GMSOFFs().load_xml(f"{PROJECT_DIR}/files/xml/water/SPCE_GMSO.xml").to_gmso_ff()
        #cation_ff_xml = forcefield_utilities.GMSOFFs().load_xml(f'{PROJECT_DIR}/files/xml/trappe/gmx-units-trappe-mie.xml').to_gmso_ff()
        cation_ff_xml = forcefield_utilities.GMSOFFs().load_xml(f'{PROJECT_DIR}/files/xml/ions/hfeBased_ree_params.xml').to_gmso_ff()
        counterion_ff_xml = forcefield_utilities.GMSOFFs().load_xml(f'{PROJECT_DIR}/files/xml/ions/hfeBased_ree_params.xml').to_gmso_ff()
        
        gmso_starting_box = from_mbuild(starting_box) 

        for i in gmso_starting_box.sites:
            if 'WAT' in i.molecule.name:
                i.molecule.name = WAT_STANDARD_NAME
                i.label = WAT_STANDARD_NAME
                i.molecule.isrigid = True

        force_field_dict = {
            WAT_STANDARD_NAME : wat_ff_xml,
            cation_coordinates.name : cation_ff_xml,
            counterion_coordinates.name : counterion_ff_xml
        }
        
        ## for i in range(3):
        ##     print('_________________________________________________________________________________________________________')
        ## 
        ## #import pdb; pdb.set_trace()
        ## print(f"type(gmso_starting_box) {type(gmso_starting_box)}") 
        ## print(f"type(wat_ff_xml) {type(wat_ff_xml)}") 
        ## #print(f"help(wat_ff_xml) {help(wat_ff_xml)}") 
        ## print(f"dir(wat_ff_xml) {dir(wat_ff_xml)}") 
        ## 
        ##  
        ## for i in range(3):
        ##     print('_________________________________________________________________________________________________________')

        apply(top=gmso_starting_box,
              forcefields=force_field_dict,
              identify_connections=True)
        
        write_gro(gmso_starting_box,filename="init.gro")
        write_top(gmso_starting_box,filename="init.top")
    
    current_lambda = ljLam_eleLam_to_initLam[round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]
    
    #shutil.copy(f'{PROJECT_DIR}/files/mdp/minim_lincs.mdp', f'{PROJECT_DIR}/workspace/{job.id}/em.mdp')
    
    parameters = {
        "integrator": "sd",
        "dt": 0.001,
        "nsteps": SMALL_EQ_STEPS,
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
        "nsteps": LONG_EQ_STEPS, # MID_EQ_STEPS,
        "output_control": SLOW_OUTPUT,
        "nstcalcenergy": NORMAL_CALC,
        "ref_t": TEMPERATURE,
        "pcoupl": "Berendsen",
        "ref_p": PRESSURE,
        "compressibility": 4.5e-5
    })   
    
    simple_mdp_writer(job,mdp_name=f'{NAME_EQ_NPT_BERENDSEN}.mdp',parameters=parameters, constraints=None,templates_dir=f'{PROJECT_DIR}/files/mdp/',template_name='NPTmdp_template.mdp') 

    parameters.update({
        "nsteps": MID_EQ_STEPS, #LONG_EQ_STEPS,
        "output_control": SLOW_OUTPUT,
        "nstcalcenergy": NORMAL_CALC,
        "pcoupl": "Parrinello-Rahman",
        "current_lambda": current_lambda,
        "molecule_of_interest":job.sp.metal,
        "nstdhdl":int(NORMAL_CALC*10)
    })   
    
    #simple_mdp_writer(job,mdp_name=f'{NAME_EQ_NPT_PARRINELLO}.mdp',parameters=parameters, constraints=None,templates_dir=f'{PROJECT_DIR}/files/mdp/',template_name='NPTmdp_template_constraints.mdp') 
    simple_mdp_writer(job,mdp_name=f'{NAME_EQ_NPT_PARRINELLO}.mdp',parameters=parameters, constraints=None,templates_dir=f'{PROJECT_DIR}/files/mdp/',template_name='free_energy_NPTmdp_template.mdp') 
     
    parameters.update({
        "nsteps": PRO_FREE_ENERGY_STEPS, # PRO_FREE_ENERGY_STEPS,
        "output_control": FAST_OUTPUT,
        "nstcalcenergy": NORMAL_CALC,
        "current_lambda": current_lambda,
        "molecule_of_interest":job.sp.metal,
        "nstdhdl":int(NORMAL_CALC*10)
    }) 
    
    simple_mdp_writer(job,mdp_name=f'{NAME_PRO_NPT_PARRINELLO}.mdp',parameters=parameters, constraints=None,templates_dir=f'{PROJECT_DIR}/files/mdp/',template_name='free_energy_NPTmdp_template.mdp') 

##  I don't do EM anymore; but I run it once simulation crashes on first run.
## @FlowProject.pre(init_written)
## @FlowProject.pre(mdp_written)
## @FlowProject.post(em_post)
## @FlowProject.operation(directives={"np": int(MID_CORES),  "ngpu": 1, "memory": 3.2, "walltime": MID_HOURS},with_job=True,cmd=True)
## def EM(job):
##     build_mdp = str(f'{GROMACS_PREFIX} grompp -f em.mdp -c init.gro -p init.top -o em.tpr -maxwarn 999')
##     run_gmx = str(f'{GROMACS_PREFIX} mdrun -nt {MID_CORES} -deffnm em')
##     run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
##     
##     return run_command

#@FlowProject.pre(em_post)
@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
#@FlowProject.pre(select_metals)
@FlowProject.post(eq_nvt_post)
@FlowProject.operation(directives={"np": int(MID_CORES),  "ngpu": 1, "memory": 3.2, "walltime": MID_HOURS},with_job=True,cmd=True)
def EQ_NVT(job):
    #build_mdp = str(f'{GROMACS_PREFIX} grompp -f {NAME_EQ_NVT}.mdp -c em.gro -p init.top -o {NAME_EQ_NVT}.tpr -maxwarn 999')
    build_mdp = str(f'{GROMACS_PREFIX} grompp -f {NAME_EQ_NVT}.mdp -c init.gro -p init.top -o {NAME_EQ_NVT}.tpr -maxwarn 999')
    run_gmx = str(f'{GROMACS_PREFIX} mdrun -nt {MID_CORES} -deffnm {NAME_EQ_NVT}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    
    return run_command

@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(eq_nvt_post)
#@FlowProject.pre(select_metals)
@FlowProject.post(eq_npt_post_beren)
@FlowProject.operation(directives={"np": int(MID_CORES),  "ngpu": 1, "memory": 3.2, "walltime": TWO_DAYS},with_job=True,cmd=True)
def EQ_NPT_BERENDSEN(job):
    build_mdp = str(f'{GROMACS_PREFIX} grompp -f {NAME_EQ_NPT_BERENDSEN}.mdp -c {NAME_EQ_NVT}.gro -p init.top -o {NAME_EQ_NPT_BERENDSEN}.tpr -maxwarn 999')
    run_gmx = str(f'{GROMACS_PREFIX} mdrun -nt {MID_CORES} -deffnm {NAME_EQ_NPT_BERENDSEN}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    
    return run_command

@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
#@FlowProject.pre(select_metals)
@FlowProject.pre(eq_npt_post_beren)
@FlowProject.post(eq_npt_post_parrin)
@FlowProject.operation(directives={"np": int(MID_CORES),  "ngpu": 1, "memory": 3.2, "walltime": TWO_DAYS},with_job=True,cmd=True)
def EQ_NPT_PARRINELLO(job):
    build_mdp = str(f'{GROMACS_PREFIX} grompp -f {NAME_EQ_NPT_PARRINELLO}.mdp -c {NAME_EQ_NPT_BERENDSEN}.gro -p init.top -o {NAME_EQ_NPT_PARRINELLO}.tpr -maxwarn 999')
    run_gmx = str(f'{GROMACS_PREFIX} mdrun -nt {MID_CORES} -deffnm {NAME_EQ_NPT_PARRINELLO}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    
    return run_command

@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
#@FlowProject.pre(select_metals)
@FlowProject.pre(eq_npt_post_parrin)
@FlowProject.post(pro_npt_post_parrin)
@FlowProject.operation(directives={"np": int(MID_CORES),  "ngpu": 1, "memory": 3.2, "walltime": TWO_DAYS},with_job=True,cmd=True)
def PRO_NPT_PARRINELLO(job):
    build_mdp = str(f'{GROMACS_PREFIX} grompp -f {NAME_PRO_NPT_PARRINELLO}.mdp -c {NAME_EQ_NPT_PARRINELLO}.gro -p init.top -o {NAME_PRO_NPT_PARRINELLO}.tpr -maxwarn 999')
    run_gmx = str(f'{GROMACS_PREFIX} mdrun -nt {MID_CORES} -deffnm {NAME_PRO_NPT_PARRINELLO}')
    run_command = str(f'{build_mdp}; sleep 2; {run_gmx}')
    
    return run_command


@FlowProject.pre(init_written)
@FlowProject.pre(mdp_written)
@FlowProject.pre(pro_npt_post_parrin)
@FlowProject.post(free_energy_bar_copied)
@FlowProject.operation(directives={"np": int(MID_CORES),  "ngpu": 0, "memory": 3.2, "walltime": MIN_HOURS},with_job=True,cmd=True) # depending how size of .xvg file this can be an intensive process.
def FREE_ENERGY_FILES_RENAMED(job):
    current_lambda = ljLam_eleLam_to_initLam[round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]
    run_command = str(f'cp {NAME_PRO_NPT_PARRINELLO}.xvg {NAME_PRO_NPT_PARRINELLO}_{current_lambda}.xvg')
    
    return run_command


@FlowProject.pre(data_collected) ## DUMMY TO AVOID THE JOB
@FlowProject.pre(free_energy_bar_copied)
@FlowProject.pre(pro_npt_post_parrin)
@FlowProject.post(data_collected)
@FlowProject.operation(directives={ "np": int(1),  "ngpu": 0, "memory": 1.1, "walltime": MIN_HOURS})
def GRAPH_AND_COLLECT_PROPERTIES(job):
    with(job):
        properties_of_interest = ["Potential", "Pressure", "Total-Energy", "Temperature", "Density"]
        properties_of_interest_to_search_string_dict = {
            # ex:
            # Disper.-corr.           : ['Disper. corr.','(kJ/mol)'] 
            # Pres-ZZ                 : ['Pres-ZZ','(bar)']
            # Total-Energy            : ['Total Energy','(kJ/mol)']
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
        
        ###### --------- Ensure the indexes correspond to properteis we need  --------- ######
        
        gromacs_input = b'1\n0\n'
        result = subprocess.run(
        [f"{GROMACS_PREFIX}", "energy", "-f", f"{NAME_PRO_NPT_PARRINELLO}.edr", "-o", "dummy_data.xvg"],
        input=gromacs_input,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT 
        )
        
        with open("gmx_energy_index_reader.txt", "wb") as f:
            f.write(result.stdout)
            
        with open("gmx_energy_index_reader.txt", "r") as f:
            text = f.read()
            
        pattern = r'\b(\d+)\s+([^\s][^ \n]*)' # r"^\s*(\d+)\s+([^\n]+?)\s{2,}"
        matches = re.findall(pattern, text, re.MULTILINE)
        
        index_map = {name.strip(): int(index) for index, name in matches}
        results = {prop: index_map[prop] for prop in properties_of_interest if prop in index_map}
        
        newline_string = "\n".join(str(results[prop]) for prop in properties_of_interest if prop in results)
        
        ###### --------- Ensure the indexes correspond to properteis we need  --------- ######
        
        p = subprocess.Popen([f'{GROMACS_PREFIX}', '-quiet', 'energy', '-f', f'{NAME_PRO_NPT_PARRINELLO}.edr', '-o', f'{GENERAL_LOCAL_DATA}_{NAME_PRO_NPT_PARRINELLO}.xvg'], stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        out,err = p.communicate(f'{newline_string}'.encode('utf-8'))#(b'13\n14\n15\n16\n17\n18\n19\n20\n0\n')
        capture = out.decode()
        
        Dummy_GMX_output = open(f'{GENERAL_LOCAL_DATA}_{NAME_PRO_NPT_PARRINELLO}.txt','w')
        Dummy_GMX_output.write(capture)
        Dummy_GMX_output.close()
        
        Dummy_GMX_output = open(f'{GENERAL_LOCAL_DATA}_{NAME_PRO_NPT_PARRINELLO}.txt','r')
        aggregate_surTenFile = open(f"../../{GENERAL_GLOBAL_DATA}.txt",'a')
        
                
        for a_single_line in Dummy_GMX_output:
            for property_str in properties_of_interest:
                search_property_str_dict = properties_of_interest_to_search_string_dict[property_str]
                search_str_start = search_property_str_dict[0]
                search_str_end = search_property_str_dict[1]
                
                if (search_str_start in a_single_line) and (search_str_end in a_single_line):
                    numpyCatcher=np.fromstring(a_single_line.strip(f'{search_str_start}{search_str_end}'),dtype=float,sep=' ')[0]
                    properties_of_interest_storage_dict[property_str] = numpyCatcher
                                    
            # read Dummy_GMX_output write to aggregate_surTenFile
            
        aggregate_surTenFile.write(f"{job.id:<42} {job.sp.lambda_LJ:<8} {job.sp.lambda_ELE:<8} {properties_of_interest_storage_dict['Potential']:<42} " 
                                  f" {properties_of_interest_storage_dict['Pressure']:<42} " 
                                  f" {properties_of_interest_storage_dict['Total-Energy']:<42} " 
                                  f" {properties_of_interest_storage_dict['Temperature']:<42} "
                                  f" {properties_of_interest_storage_dict['Density']:<42} \n")
        
        
        
        ### graph the data of simulation to see if it's converged

        Dummy_GMX_output.close(); aggregate_surTenFile.close()
        
        xvg_png_datasource = open(f'{GENERAL_LOCAL_DATA}_{NAME_PRO_NPT_PARRINELLO}.xvg','r')
        #lines = xvg_png_datasource.strip().split('\n')
        lines = xvg_png_datasource.readlines()
        
        header_lines = []
        data_lines = []
        in_data_section = False
        
        
        for line in lines:
            if line.startswith('@') or line.startswith('#'):
                header_lines.append(line)
                if "@TYPE xy" in line:
                    in_data_section = True
            else : # in_data_section # #and line.strip(): #and not line.startswith('#'):
                data_lines.append(line.strip())

        # Extract column names from header
        column_names = {}
        xaxis_label = "Time (ps)" # Default, will be updated if found
        yaxis_label = ""
        title = ""

        for line in header_lines:
            if line.startswith('@ s'):
                match = re.search(r'@ s(\d+) legend "(.+)"', line)
                if match:
                    col_index = int(match.group(1))
                    legend_name = match.group(2)
                    column_names[col_index] = legend_name
            elif line.startswith('@ xaxis'):
                match = re.search(r'@ xaxis\s+label "(.+)"', line)
                if match:
                    xaxis_label = match.group(1)
            elif line.startswith('@ yaxis'):
                match = re.search(r'@ yaxis\s+label "(.+)"', line)
                if match:
                    yaxis_label = match.group(1)
            elif line.startswith('@ title'):
                match = re.search(r'@\s+title "(.+)"', line)
                if match:
                    title = match.group(1)

        # Create a list of column names in order
        # The first column is always the x-axis label.
        # Then append the other column names based on their index.
        ordered_column_names = [xaxis_label]
        max_col_index = max(column_names.keys()) if column_names else -1
        for i in range(max_col_index + 1):
            if i in column_names:
                ordered_column_names.append(column_names[i])

        # Read data into a pandas DataFrame
        # Using io.StringIO to treat the list of data lines as a file
        #print(f'data_lines: {data_lines}')
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=r'\s+', header=None)

        # Assign column names to the DataFrame
        df.columns = ordered_column_names[:len(df.columns)]

        # Plotting the data
        num_cols = len(df.columns) - 1  # Exclude the 'Time (ps)' column
        fig, axes = plt.subplots(num_cols, 1, figsize=(10, 5 * num_cols), sharex=True)

        if num_cols == 1:
            axes = [axes] # Ensure axes is iterable even for a single subplot
        
        for i, col_name in enumerate(df.columns[1:]): # Iterate over data columns, skipping time
            axes[i].plot(df[xaxis_label], df[col_name])
            axes[i].set_ylabel(f'{col_name} {yaxis_label}') # Add yaxis_label to each subplot's y-label
            axes[i].grid(True)
            #axes[i].set_title(f'{title}: {col_name}') # <- from chat
            key_to_mean_data = ''
            for key, value_list in properties_of_interest_to_search_string_dict.items(): 
                if col_name in value_list[0]:
                    key_to_mean_data = key
            
            axes[i].set_title(f'{col_name}; mean {properties_of_interest_storage_dict[key_to_mean_data]}') # <- hooman edited.

        axes[-1].set_xlabel(xaxis_label) # Set x-label only on the last subplot

        plt.tight_layout()
        plt.savefig(f'{GENERAL_LOCAL_DATA}_{NAME_PRO_NPT_PARRINELLO}.png')
        plt.close()
        
        
        ## legends = [""] * 5
        ## 
        ## with open(xvg_png_datasource, "r") as f:
        ##     for line in f:
        ##         if line.startswith("@"):
        ##             match = re.match(r'@ s(\d+) legend "(.*)"', line)
        ##             if match:
        ##                 series_index = int(match.group(1)) + 1  # s0 -> column 1 (time is 0)
        ##                 if series_index < len(legends):
        ##                     legends[series_index] = match.group(2)
        ##         elif not line.startswith(("#", "@")):
        ##             break  # Stop after headers
        ##         
        ## ## df = pd.read_csv(
        ## ##     xvg_png_datasource,
        ## ##     delim_whitespace=True,
        ## ##     comment="#",
        ## ##     names=["Time", "Y1", "Y2", "Y3", "Y4"],
        ## ##     usecols=[0, 1, 2, 3, 4]
        ## ## )
        ## 
        ## df = pd.read_csv(
        ##     xvg_png_datasource,
        ##     sep=r"\s+",
        ##     comment="#",
        ##     names=["Time", "Y1", "Y2", "Y3", "Y4"],
        ##     usecols=[0, 1, 2, 3, 4],
        ##     engine="python"  # Needed for regex separator
        ## )
        ## 
        ## df = df.apply(pd.to_numeric, errors='coerce')  # convert all to floats, set non-numerics to NaN
        ## df = df.dropna()  # drop any rows with NaN
 
        ## 
        ## fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        ## 
        ## 
        ## for i in range(4):
        ##     axes[i].plot(df["Time"], df.iloc[:, i + 1], color="black")
        ##     axes[i].set_title(legends[i + 1] or f"Y{i+1}", color="black")
        ##     axes[i].tick_params(axis='x', colors='black')
        ##     axes[i].tick_params(axis='y', colors='black')
        ##     
        ## axes[-1].set_xlabel("Time", color="black")
        ## plt.tight_layout()
        ## plt.savefig(f'{GENERAL_LOCAL_DATA}_{NAME_PRO_NPT_PARRINELLO}.txt', dpi=300)
        
        

if __name__ == '__main__':
    FlowProject().main()
    
