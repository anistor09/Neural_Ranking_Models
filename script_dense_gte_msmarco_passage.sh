#!/bin/bash

#SBATCH --job-name="test_library"
#SBATCH --time=01:40:00
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load python
module load py-pip
module load openjdk
module load cuda


# Install dependencies
python -m pip install --user python-terrier==0.10.0 fast-forward-indexes==0.2.0 jupyter ipywidgets transformers typing pathlib

# Run the experiment
srun python -m gte_base_en_v1_5.dense_INDEXERS.dense_index_one_dataset_msmarco_passage > prints.txt
