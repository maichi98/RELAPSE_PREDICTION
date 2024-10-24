#!/bin/bash

# Ensure that the script exits if a command fails
set -e

# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_relapse_prediction

# Define the project directory:
DIR_PROJECT="/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION/relapse_prediction"

# Run Features generation :
#python "$DIR_PROJECT/features/features.py" --mp --num_workers 4

# Run Labels generation :
#python "$DIR_PROJECT/labels/labels.py"  --mp --num_workers 4

# ROC for cercare features :
#python "$DIR_PROJECT/roc/cercare_roc.py"  --mp --num_workers 2 --feature



# ROC for MRI features :
#python "$DIR_PROJECT/roc/mri_roc.py" --mp --num_workers 1



# cercare total ROC features :
#python "$DIR_PROJECT/total_roc/cercare_total_roc.py"

# MRI total ROC features :
#python "$DIR_PROJECT/total_roc/mri_total_roc.py"

# Deactivate the conda environment
conda deactivate
