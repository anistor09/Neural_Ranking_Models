#!/bin/bash

#SBATCH --job-name="cpu_metrics_snowflake"
#SBATCH --time=05:00:00
#SBATCH --partition=compute-p2
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
python -m pip install --user func-timeout



# Run the experiment
srun python -m snowflake.experiments.ranking_measures_snowflake_m > ranking_measures_snowflake_status.txt
