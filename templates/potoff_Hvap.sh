{% extends "slurm.sh" %}

{% block header %}
{{- super () -}}
{% set gpus = operations|map(attribute='directives.ngpu')|sum %}
{% set cpus = operations|map(attribute='directives.np')|sum %}

{% if gpus %}
#SBATCH -N 1
#SBATCH --exclude=ressrv7ai8111,reslab32ai8111,ressrv10ai8111,ressrv8ai8111,ressrv13ai8111,ressrv14ai8111,ressrv15ai8111,ressrv12ai8111

{%- else %}
#SBATCH -N 1
#SBATCH --exclude=res-lab34-ai8111,res-lab41-ai8111,ressrv4ai8111,ressrv6ai8111,res-lab43-ai8111,res-lab33-ai8111,reslab44ai8111
{%- endif %}


echo  "Running on host" 
hostname
echo  "Time is" date
date

source /home6/go0719/.bashrc
export PATH=$PATH:/usr/local/gromacs/bin
export GMX_MAXCONSTRWARN=-1
#conda init bash
conda activate mosdef-study38

{% endblock header %}

{% block body %}
    {{- super () -}}

{% endblock body %}
