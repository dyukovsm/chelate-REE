{% extends "slurm.sh" %}
{% block header %}
{% set gpus = operations|map(attribute='directives.ngpu')|sum %}
    {{- super () -}}

{% if gpus %}
#SBATCH --gpus-per-node=1
######SBATCH --ntasks-per-gpu=1
{%- endif %}

{% set walltime = operations |calc_walltime(parallel) %}
{% if walltime %}
#SBATCH --time {{walltime|format_timedelta}}
{% endif %}


{% block tasks %}
#SBATCH --ntasks={{operations|calc_tasks('np',parallel, force) }}
{% endblock tasks %}

#SBATCH -N 1
#SBATCH --cpus-per-task=1
######SBATCH --nodelist=res-lab42-ai8111
######SBATCH --ntasks=1
######SBATCH --ntasks-per-core=1
######SBATCH --ntasks-per-node={{operations|calc_tasks('np',parallel, force) }}

echo  "Running on host" $HOSTNAME
current_time=$(date +"%Y-%m-%d %H:%M:%S")
echo  "Time is" $current_time
source /home6/go0719/.bashrc
mamba activate hybrid_amber_mosdef

{% if gpus %}
#module load cuda/11.0
{%- endif %}
{% endblock header %}
{% block body %}
    {{- super () -}}
{% endblock body %} (edited) 

