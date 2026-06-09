import signac
import numpy as np
import os

project = signac.init_project()


# metal = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Er','Tm','Lu'] # to test
metal = ['Nd']
replicate = [0] # , 1, 2]
lambda_LJ   = [1.0] # [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
lambda_ELE  = [1.0] # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

total_statepoints = list()
legend = open('legend.txt','w')
legend.write('job \t sp \n')
print('job \t sp')

for i in range(len(metal)):
    for j in range(len(replicate)):
        for k in range(len(lambda_ELE)):
            if lambda_ELE[k] == lambda_ELE[-1]:
                for l in range(len(lambda_LJ)):
                    statepoint = {
                        "metal": metal[i],
                        "replicate": replicate[j],
                        "lambda_LJ": lambda_LJ[l],
                        "lambda_ELE": lambda_ELE[k]
                    }
                    total_statepoints.append(statepoint)
            else:
                statepoint = {
                    "metal": metal[i],
                    "replicate": replicate[j],
                    "lambda_LJ": lambda_LJ[0],
                    "lambda_ELE": lambda_ELE[k]
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

    
    
