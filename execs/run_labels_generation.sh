#!/bin/bash

# Ensure that the script exits if a command fails
set -e

# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_relapse_prediction

# Define the project directory:
DIR_PROJECT="/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION/relapse_prediction"

# Run Labels generation  :
python "$DIR_PROJECT/labels/labels.py"  --mp --num_workers 4

# Add the new Labels :
python "$DIR_PROJECT/labels/add_new_labels.py" --mp --num_workers 4

# Deactivate the conda environment
conda deactivate
