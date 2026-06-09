{% extends "slurm.sh" %}
{% block header %}
{% set gpus = operations|map(attribute='directives.ngpu')|sum %}
    {{- super () -}}
{% if gpus %}
#SBATCH  --gpus-per-node=1
#SBATCH --ntasks-per-gpu=1
{%- else %}
{%- endif %}
#SBATCH -N 1
#SBATCH --ntasks-per-core=1
#SBATCH --exclude=ressrv7ai8111, reslab32ai8111
echo  "Running on host" hostname
echo  "Time is" date
conda activate works_well
#module load python/3.9
#module swap gnu7 intel/2019
{% if gpus %}
#module load cuda/11.0
{%- endif %}
{% endblock header %}
{% block body %}
    {{- super () -}}
{% endblock body %}