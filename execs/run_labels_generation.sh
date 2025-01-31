#!/bin/bash

# Ensure that the script exits if a command fails
set -e

# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_relapse_prediction

# Define the project directory:
DIR_PROJECT="/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION/relapse_prediction"

# Run Labels generation  for reg_tp = Affine:
#python "$DIR_PROJECT/labels/labels.py" --reg_tp "Affine" --mp --num_workers 4

# Run Labels generation  for reg_tp = SyN:
python "$DIR_PROJECT/labels/labels.py" --reg_tp "SyN" --mp --num_workers 8

# Deactivate the conda environment
conda deactivate