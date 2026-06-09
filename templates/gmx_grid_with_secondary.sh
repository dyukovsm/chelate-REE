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
#SBATCH -q secondary
#SBATCH -N 1
module unload openmpi3
#module swap gnu9/9.1.0 gnu7/7.3.0
module unload gnu7
module load gnu9/9.1.0
module load gromacs/2022

{%- endif %}

hostname
date

source ~/.bashrc
module load python/3.10
conda activate works_well_noleak



{% endblock header %}

{% block body %}
    {{- super () -}}


{% endblock body %}
