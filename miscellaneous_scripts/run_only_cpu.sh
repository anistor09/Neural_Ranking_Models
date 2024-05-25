#!/bin/bash

#SBATCH --job-name="cpu_regular"
#SBATCH --time=00:20:00
#SBATCH --partition=compute-p2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load python
module load py-pip
module load openjdk
module load cuda


# Install dependencies
python -m pip install --user python-terrier==0.10.0 fast-forward-indexes==0.2.0 jupyter ipywidgets transformers typing pathlib

# Run the experiment
srun python test1000docs_per_query_disk.py > prints.txt

