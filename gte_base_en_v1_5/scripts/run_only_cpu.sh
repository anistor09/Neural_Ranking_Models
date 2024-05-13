#!/bin/bash

#SBATCH --job-name="test_library"
#SBATCH --time=00:30:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load python
module load py-pip
module load openjdk
module load cuda


# Install dependencies
python -m pip install --user python-terrier==0.10.0 fast-forward-indexes==0.2.0 jupyter ipywidgets transformers typing pathlib

# Run the experiment
srun python run_python_cell.py > prints.txt

