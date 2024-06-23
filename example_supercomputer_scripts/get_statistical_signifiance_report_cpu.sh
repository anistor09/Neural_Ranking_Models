#!/bin/bash

#SBATCH --job-name="find_statistical_signifiance"
#SBATCH --time=15:00:00
#SBATCH --partition=compute-p2 # GPU is not needed anything runs with CUDA
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G # High memory usage, it fails with 16 G RAM on local machine
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load python
module load py-pip
module load openjdk
module load cuda


export IR_DATASETS_SKIP_DISK_FREE=true
export IR_DATASETS_HOME=/scratch/anistor/.ir_datasets/


# Install dependencies
python -m pip install --user python-terrier==0.10.0 fast-forward-indexes==0.2.0 jupyter ipywidgets transformers typing pathlib func-timeout ranx



# Run the experiment
srun python -m statistical_significance.find_statistical_signifiance_script > find_statistical_signifiance_script_status.txt

