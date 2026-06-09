{% extends "slurm.sh" %}

{% block header %}
    {{- super () -}}
    #SBATCH -q secondary
    #SBATCH --time=300:00:00
    #SBATCH -N 1
    #SBATCH -n 4
    #SBATCH --mem=8G
    #SBATCH --ntasks-per-core=1
    
    ml unload gnu7
    ml gnu9
    
    
    #SBATCH -o output-%j.dat
    #SBATCH -e error-%j.dat
    
    hostname
    date
    
    source ~/.bashrc
    conda activate mosdef-study38
    
    module load python/3.8

{% endblock header %}

{% block body %}
    {{- super () -}}


{% endblock body %}
