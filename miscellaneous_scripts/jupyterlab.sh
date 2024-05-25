#!/bin/bash


#SBATCH --job-name="jupyter"
#SBATCH --time=00:20:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

# Load modules:
module load 2022r2
module load openmpi
module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate jupyterlab
cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888
conda deactivate