{% extends "slurm.sh" %}

{% block header %}
{{- super () -}}
{% set gpus = operations|map(attribute='directives.ngpu')|sum %}
{% set cpus = operations|map(attribute='directives.np')|sum %}

{% if gpus %}
#SBATCH -N 1
#SBATCH --gres=gpu:1 
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --ntasks-per-core=1
#SBATCH --exclude=reslab35ai8111

{%- else %}
#SBATCH -N 1
#SBATCH --ntasks-per-core=1
#SBATCH --exclude=reslab35ai8111,reslab32ai8111

{%- endif %}


echo  "Running on host" 
hostname
echo  "Time is" date
date

source /home6/go0719/.bashrc
source /home6/go0719/.conda/envs/works_well_noleak/bin/python3.10
export PATH=$PATH:/usr/local/gromacs/bin
export GMX_MAXCONSTRWARN=-1
#conda init bash
conda activate works_well_noleak

{% endblock header %}

{% block body %}
    {{- super () -}}

{% endblock body %}
