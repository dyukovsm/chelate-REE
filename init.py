# pyrefly: ignore [missing-import]
import signac
# pyrefly: ignore [missing-import]
import numpy as np
import os
from files.python_files import names

project = signac.init_project()

local_eleLam_ljLam_to_initLam = names.eleLam_ljLam_to_initLam
lambda_BONDED = sorted({bonded for bonded, ele, lj in local_eleLam_ljLam_to_initLam})
lambda_ELE = sorted({ele for bonded, ele, lj in local_eleLam_ljLam_to_initLam})
lambda_LJ  = sorted({lj for bonded, ele, lj in local_eleLam_ljLam_to_initLam})
# metal = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Er','Tm','Lu','Al','Fe','Ca'] # to test
metal = ['Al','Fe','Ca','La','Ce','Nd','Eu','Dy','Lu'] # to test
polypeptide = ['LBT5-'] #['LBT5-','LBT3-','DUM3+']
replicate = [0] # , 1, 2]
unNested_usesTemplates = False # True


total_statepoints = list()
legend = open('legend.txt','w')
legend.write('job \t sp \n')
print('job \t sp')

for i in range(len(metal)):
    for j in range(len(replicate)):
        for n in range(len(lambda_BONDED)):
            if lambda_BONDED[n] == lambda_BONDED[-1]:
                # Bonded is at max, now iterate through ELE
                for k in range(len(lambda_ELE)):
                    if lambda_ELE[k] == lambda_ELE[-1]:
                        # ELE is at max, now iterate through LJ
                        for m in range(len(polypeptide)):
                            for l in range(len(lambda_LJ)):
                                statepoint = {
                                    "metal": metal[i],
                                    "replicate": replicate[j],
                                    "lambda_BONDED": lambda_BONDED[n],
                                    "lambda_ELE": lambda_ELE[k],
                                    "lambda_LJ": lambda_LJ[l],
                                    "polypeptide": polypeptide[m],
                                    "unNested_usesTemplates": unNested_usesTemplates
                                }
                                total_statepoints.append(statepoint)
                    else:
                        # ELE not at max, use LJ[0]
                        for m in range(len(polypeptide)):
                            statepoint = {
                                "metal": metal[i],
                                "replicate": replicate[j],
                                "lambda_BONDED": lambda_BONDED[n],
                                "lambda_ELE": lambda_ELE[k],
                                "lambda_LJ": lambda_LJ[0],
                                "polypeptide": polypeptide[m],
                                "unNested_usesTemplates": unNested_usesTemplates
                            }
                            total_statepoints.append(statepoint)
            else:
                # Bonded not at max, use ELE[0] and LJ[0]
                for m in range(len(polypeptide)):
                    statepoint = {
                        "metal": metal[i],
                        "replicate": replicate[j],
                        "lambda_BONDED": lambda_BONDED[n],
                        "lambda_ELE": lambda_ELE[0],
                        "lambda_LJ": lambda_LJ[0],
                        "polypeptide": polypeptide[m],
                        "unNested_usesTemplates": unNested_usesTemplates
                    }
                    total_statepoints.append(statepoint)

## for i in range(len(metal)):
##     for j in range(len(replicate)):
##         for k in range(len(lambda_LJ)):
##             if lambda_LJ[k] == lambda_LJ[-1]:
##                 for l in range(len(lambda_ELE)):
##                     statepoint = {
##                         "metal": metal[i],
##                         "replicate": replicate[j],
##                         "lambda_LJ": lambda_LJ[k],
##                         "lambda_ELE": lambda_ELE[l]
##                     }
##                     total_statepoints.append(statepoint)
##             else:
##                 statepoint = {
##                     "metal": metal[i],
##                     "replicate": replicate[j],
##                     "lambda_LJ": lambda_LJ[k],
##                     "lambda_ELE": lambda_ELE[0]
##                 }
##                 total_statepoints.append(statepoint)
            
        


for sp in total_statepoints:
    job=project.open_job(
        statepoint=sp,
    ).init()
    legend.write(f' {job} \t\t {sp}\n')
    print(f'{job} \t\t {sp}')
 
 
legend.close()

    
    
