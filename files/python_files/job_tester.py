import math
#import mbuild as mb
#from foyer import Forcefield
#import foyer
import pandas as pd
import numpy as np
import random
import time
#from parmed import residue
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
#import parmed
#from parmed import gromacs
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

def runWithTemplateAbsent(job):
    test_passed = False
    if not job.sp.unNested_usesTemplates:
        test_passed = True
    return test_passed


import os

def gmx_log_finished(job, log_filename, tail_bytes=30000):
    """
    Check if GROMACS log file indicates successful completion.
    Optimized for high-throughput workflow queries via byte-seeking.
    """
    with job:
        if not job.isfile(log_filename):
            return False
            
        # Read strictly the tail end of the file in binary mode
        with open(log_filename, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            
            if file_size == 0:
                return False
                
            # Seek back from the end (or from the start if it's a tiny file)
            seek_pos = max(0, file_size - tail_bytes)
            f.seek(seek_pos, os.SEEK_SET)
            
            # Load the chunk as a single byte-string
            tail_data = f.read()
            
    # Perform C-level substring checks directly on the byte block.
    # We check for the completion string first.
    if b"Finished mdrun on " in tail_data:
        # If it finished, ensure it wasn't a graceful shutdown from an interrupt
        if b"stopping within" not in tail_data:
            return True
            
    return False

############################__BUILD_JOBS__############################


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

    
    
##################################################################################

# --- Active Workflow Condition & Label Functions ---

@FlowProject.label
def pre_equilibrated(job):
    with job:
        test_passed = False
        if job.isfile(f"{names.NAME_PRE_EQ_NPT_BERENDSEN}.gro"):
            with open(f"{names.NAME_PRE_EQ_NPT_BERENDSEN}.gro", "r") as file_with_lines:
                lines = file_with_lines.readlines()
            for single_line in lines:
                if job.sp.metal in single_line:
                    test_passed = True
                    break
    return test_passed


def templatedOrEquilibrated_eqNVT(job):
    """Check if job uses pre-equilibrated template OR has completed NPT equilibration."""
    if job.sp.unNested_usesTemplates:
        return pre_equilibrated(job)
    else:
        return gmx_log_finished(job, f"{names.NAME_EQ_NVT}.log")

def templatedOrEquilibrated_eqNPT(job):
    """Check if job uses pre-equilibrated template OR has completed NPT equilibration."""
    if job.sp.unNested_usesTemplates:
        return pre_equilibrated(job)
    else:
        return gmx_log_finished(job, f"{names.NAME_EQ_NPT_BERENDSEN}.log")



@FlowProject.label
def eq_nvt_post(job):
    return gmx_log_finished(job, f"{names.NAME_EQ_NVT}.log")


@FlowProject.label
def eq_npt_post_beren(job):
    return gmx_log_finished(job, f"{names.NAME_EQ_NPT_BERENDSEN}.log")


@FlowProject.label
def eq_canon_post(job):
    return gmx_log_finished(job, f"{names.NAME_EQ_CANON}.log")


@FlowProject.label
def pro_canon_post(job):
    return gmx_log_finished(job, f"{names.NAME_PRO_CANON}.log")


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

def xvg_present_for_all(*jobs):
    test_passed = True
    for job in jobs:
        with job:
            if job.isfile(f'{names.NAME_PRO_CANON}.xvg'):
                current_lambda = names.eleLam_ljLam_to_initLam[round(job.sp.lambda_ELE, 5), round(job.sp.lambda_LJ, 5)]
                if not job.isfile(f'{names.NAME_PRO_CANON}_{current_lambda}.xvg'):
                    test_passed = False
                    break
            else:
                test_passed = False
                break
    return test_passed

def aggregated_data_present(*jobs):
    test_passed = False
    group_parts = [str(jobs[0].sp.metal)]
    if getattr(jobs[0].sp, 'polypeptide', None):
        group_parts.append(str(jobs[0].sp.polypeptide))
    group_parts.append(str(jobs[0].sp.replicate))
    group_parts.append(str(jobs[0].sp.unNested_usesTemplates))
    group_name = "_".join(group_parts)
    target_dir = os.path.join(names.PROJECT_DIR, names.ANALYSIS_DIR_PREFIX, group_name)
    output_file = os.path.join(names.PROJECT_DIR, "aggregated_free_energy.txt")
    
    if os.path.exists(target_dir):
        if os.path.exists(os.path.join(target_dir, f"{names.GENERAL_FILE_PREFIX}.txt")):
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    if group_name in f.read():
                        test_passed = True
    return test_passed

@FlowProject.label
def rdf_calculated(job):
    group_parts = [str(job.sp.metal)]
    if getattr(job.sp, 'polypeptide', None):
        group_parts.append(str(job.sp.polypeptide))
    group_parts.append(str(job.sp.replicate))
    group_parts.append(str(job.sp.unNested_usesTemplates))
    group_name = "_".join(group_parts)
    target_dir = os.path.join(names.PROJECT_DIR, names.ANALYSIS_DIR_PREFIX, group_name)
    output_file = os.path.join(target_dir, "rdf_analysis.txt")
    
    if not os.path.exists(output_file):
        return False
    with open(output_file, 'r') as f:
        contents = f.read()
        if job.id in contents:
            return True
    return False
