import math
import mbuild as mb
from foyer import Forcefield
import foyer
import pandas as pd
import numpy as np
import random
import time
from parmed import residue
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import parmed
from parmed import gromacs
import signac
from flow import FlowProject, aggregator
from flow.environment import DefaultSlurmEnvironment
import flow
import subprocess
import os
from jinja2 import Environment, FileSystemLoader
import shutil
from files.python_files import names


extension_list_of_common_files = [".gro", ".trr", ".log", ".edr", ".tpr"]   # former extension_list_list
extension_list_inits = [".gro",".top"]


def return_file_with_extensions(file_names,extension_list):
    file_names_with_extensions = [name + ext for name in file_names for ext in extension_list]

    return file_names_with_extensions


def test_existence_simple(job,file_lsit):
    with(job):
        test_passed = False
        for i in file_lsit:
            if job.isfile(i):
                test_passed = True
            elif not(job.isfile(i)):
                test_passed = False
                break
        return test_passed
    
    
def look_in_file(job,file_list,look_string,debug=False,check_for_not=False,check_for_not_str=''):
    with(job):
        test_passed = False
        if debug:
            missing_file = open(f"debug_look_IN_file_{file_list[0]}.txt",'w')
            missing_file.write('test')
            missing_file.close()
            #close the file before
            missing_file = open(f"debug_look_IN_file_{file_list[0]}.txt",'a')
            #for i in file_list:
            #    missing_file.write(f'{i}\n')
        for i in file_list:
            if job.isfile(i):
                file_with_lines = open(f'{i}','r')
                lines = file_with_lines.readlines()
                for j in lines:
                    if debug:
                        if look_string not in j:
                            missing_file.write(f'{look_string} not found in \t\t\t {j}\n')
                        else:
                            missing_file.write(f'{look_string} WAS FOUND in \t\t\t {j}\n')
                    if check_for_not:
                        if check_for_not_str in j:
                            test_passed = False
                            break
                        
                    if look_string in j:
                        test_passed = True
                        break
                file_with_lines.close()
            elif debug:
                missing_file.write(f'ERROR {i} not found.\n')
    return test_passed


def run_only_one(job):
    test_passed = False
    if job.sp.replica < 1:
        test_passed = True
    return test_passed


def important_jobs(job):
    test_passed = False
    if job.sp.replicas < 1:
        #if 'PME' in job.sp.cut_type:
            test_passed = True
    return test_passed

############################__JOB_SPECIFIC_FUNCTIONS__############################


@FlowProject.label
def inits_written(job):
    with(job):
        check_these_files = return_file_with_extensions(file_names=['init'],extension_list = extension_list_inits)
        return test_existence_simple(job,check_these_files)
    

@FlowProject.label
def mdps_written(job):
    with(job):
        check_these_files = return_file_with_extensions(file_names=[names.NAME_EQ_NVT,
                                                                    names.NAME_EQ_SURFTEN,names.NAME_PRO_SURFTEN],extension_list = ['.mdp'])
        return test_existence_simple(job,check_these_files)
    
    
@FlowProject.label
def build_input_starter(job):
    with(job):
        starter_bool = not(inits_written(job)) and not(mdps_written(job))
        return starter_bool
    
    
##################################################################################

# --- Active Workflow Condition & Label Functions ---

@FlowProject.label
def init_written(job):
    with job:
        test_passed = False
        for i in names.INIT_FILE_LIST:
            if job.isfile(i):
                test_passed = True
            else:
                test_passed = False
                break
    return test_passed


@FlowProject.label
def mdp_written(job):
    with job:
        test_passed = False
        for i in names.MDP_FILE_LIST:
            if job.isfile(i):
                test_passed = True
            else:
                test_passed = False
                break
    return test_passed


@FlowProject.label
def select_metals(job):
    with job:
        test_passed = False
        metals_to_run = ['Fe', 'Gd', 'Hf']
        for metal in metals_to_run:
            if job.sp.metal == metal:
                test_passed = True
                break
    return test_passed


@FlowProject.label
def eq_nvt_post(job):
    with job:
        test_passed = False
        if job.isfile(f"{names.NAME_EQ_NVT}.log"):
            with open(f"{names.NAME_EQ_NVT}.log", "r") as file_with_lines:
                lines = file_with_lines.readlines()
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
    with job:
        test_passed = False
        if job.isfile(f"{names.NAME_EQ_NPT_BERENDSEN}.log"):
            with open(f"{names.NAME_EQ_NPT_BERENDSEN}.log", "r") as file_with_lines:
                lines = file_with_lines.readlines()
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
    with job:
        test_passed = False
        if job.isfile(f"{names.NAME_EQ_CANON}.log"):
            with open(f"{names.NAME_EQ_CANON}.log", "r") as file_with_lines:
                lines = file_with_lines.readlines()
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
    with job:
        test_passed = False
        if job.isfile(f"{names.NAME_PRO_CANON}.log"):
            with open(f"{names.NAME_PRO_CANON}.log", "r") as file_with_lines:
                lines = file_with_lines.readlines()
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
    with job:
        test_passed = False
        if job.isfile(f'{names.NAME_PRO_CANON}.xvg'):
            current_lambda = names.eleLam_ljLam_to_initLam[round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]
            if job.isfile(f'{names.NAME_PRO_CANON}_{current_lambda}.xvg'):
                test_passed = True
    return test_passed


@FlowProject.label
def data_collected(job):
    test_passed = False
    local_name_of_file = f'{names.GENERAL_GLOBAL_DATA}.txt'
    if os.path.exists(local_name_of_file):
        with open(local_name_of_file, "r") as f:
            contents = f.read()
            if job.id in contents:
                test_passed = True
    return test_passed
