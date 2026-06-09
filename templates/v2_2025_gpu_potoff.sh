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
#####SBATCH --nodelist=reslab35ai8111,res-lab33-ai8111 # does not work with more than 1 node
#SBATCH --exclude=ressrv8ai8111,ressrv10ai8111,ressrv12ai8111,ressrv13ai8111,ressrv14ai8111,res-lab34-ai8111,res-lab43-ai8111,ressrv8ai8111,ressrv10ai8111,ressrv13ai8111,res-lab41-ai8111,res-lab42-ai8111,reslab44ai8111,ressrv4ai8111,ressrv6ai8111,ressrv12ai8111,reslab32ai8111
#SBATCH --exclusive # nuclear option
#SBATCH --sockets-per-node=1
#####SBATCH --ntasks-per-socket=1 # {{operations|calc_tasks('np',parallel, force) }}  # --sockets-per-node=1
#####SBATCH --threads-per-core=2
#####SBATCH -B *:1:1    # tried to substitute for --cpu-bind
echo  "Running on host" $HOSTNAME
echo  "Running on GOMP_CPU_AFFINITY" $GOMP_CPU_AFFINITY
echo  "Running on KMP_AFFINITY " $KMP_AFFINITY 
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

