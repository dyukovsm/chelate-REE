{% extends "slurm.sh" %}
{% block header %}
{% set gpus = operations|map(attribute='directives.ngpu')|sum %}
    {{- super () -}}

{% if gpus %}
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-gpu=1
{%- endif %}

{% set walltime = operations |calc_walltime(parallel) %}
{% if walltime %}
#SBATCH --time {{walltime|format_timedelta}}
{% endif %}

{% block tasks %}
#SBATCH --cpus-per-task={{operations|calc_tasks('np',parallel, force) }}
{% endblock tasks %}

#SBATCH -N 1
#SBATCH --ntasks-per-core=1
##SBATCH --nodelist=res-lab42-ai8111 
#SBATCH --exclude=ressrv8ai8111,ressrv10ai8111,res-lab34-ai8111,res-lab33-ai8111
echo  "Running on host" $HOSTNAME
echo  "Time is" date
source /home6/go0719/.bashrc
mamba activate hybrid_amber_mosdef

{% if gpus %}
#module load cuda/11.0
{%- endif %}
{% endblock header %}
{% block body %}
    {{- super () -}}
{% endblock body %} (edited) 

