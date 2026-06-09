{% extends "slurm.sh" %}

{% block header %}
    {{- super () -}}

    #SBATCH -q gpu
    #SBATCH -N 1
    #SBATCH -n 4
    #SBATCH --mem=8G  
    #SBATCH -o output-%j.dat
    #SBATCH -e error-%j.dat
    #SBATCH --constraint=avx512
    #SBATCH -t 12:0:0
    
    hostname
    date
    
    source ~/.bashrc
    conda activate mosdef-study38
    
    module load python/3.8
    
    ml unload openmpi3
    ml unload gnu7
    ml gnu9
    ml gromacs
    ml cuda

{% endblock header %}

{% block body %}
    {{- super () -}}


{% endblock body %}
