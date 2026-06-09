{% extends "slurm.sh" %}

{% block header %}
{{- super () -}}
{% set gpus = operations|map(attribute='directives.ngpu')|sum %}
{% set cpus = operations|map(attribute='directives.np')|sum %}

{% if gpus %}
#SBATCH -q gpu
#SBATCH -N 1
#SBATCH --constraint=avx512
#SBATCH --gres=gpu:1
ml unload openmpi3
module swap gnu9/9.1.0
ml unload gnu7
ml gnu9/9.1.0
ml cuda
ml gromacs/2022

{%- else %}
#SBATCH -q primary
#SBATCH -N 1
#SBATCH --constraint=avx512         
#SBATCH --exclude=sha1         
#module swap gnu9/9.1.0 gnu7/7.3.0  # old
module unload openmpi3             # disable for vmd enable for gmx    
module unload gnu7                 # disable for vmd enable for gmx
module load gnu9/9.1.0             # disable for vmd enable for gmx
module load gromacs/2022           # disable for vmd enable for gmx
#module load gnu7                    # enable for vmd disable for gmx
#module load vmd                     # enable for vmd disable for gmx

{%- endif %}

hostname
date

source ~/.bashrc
module load python/3.10
mamba activate tessellation_env



{% endblock header %}

{% block body %}
    {{- super () -}}


{% endblock body %}
