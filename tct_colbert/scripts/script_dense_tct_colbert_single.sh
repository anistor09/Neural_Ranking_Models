#!/bin/bash

#SBATCH --job-name="jupyter_test"
#SBATCH --time=10:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load python
module load py-pip
module load openjdk
module load cuda


export IR_DATASETS_SKIP_DISK_FREE=true
export IR_DATASETS_HOME=/scratch/anistor/.ir_datasets/


# Install dependencies
python -m pip install --user python-terrier==0.10.0 fast-forward-indexes==0.2.0 jupyter ipywidgets transformers typing pathlib



# Run the experiment
srun python -m tct_colbert.dense_indexers.dense_index_one_dataset > prints_tct_colbert_single.txt

