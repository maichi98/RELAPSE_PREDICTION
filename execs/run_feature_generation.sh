#!/bin/bash

# Ensure that the script exits if a command fails
set -e

# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_relapse_prediction

# Define the project directory:
DIR_PROJECT="/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION/relapse_prediction"

# Run Features generation for 
#python "$DIR_PROJECT/features/features.py" --mp --num_workers 4

# Run Labels generation :
#python "$DIR_PROJECT/labels/labels.py"  --mp --num_workers 4

# Add the new Labels :
#python "$DIR_PROJECT/labels/add_new_labels.py" --mp --num_workers 4

# ROC for cercare features :
#python "$DIR_PROJECT/roc/cercare_roc.py" --mp  --num_workers 3 --labels "L3 + L3R" "L3 + L3R_5x5x5"

# ROC for cercare features :
#python "$DIR_PROJECT/roc/cercare_roc.py"  --feature "mean_5x5x5" --mp  --num_workers 3 --labels "L3 + L3R" "L3 + L3R_5x5x5"

# ROC for MRI features :
#python "$DIR_PROJECT/roc/mri_roc.py" --mp  --num_workers 3 --labels "L3 + L3R" "L3 + L3R_5x5x5"

# ROC for MRI features :
#python "$DIR_PROJECT/roc/mri_roc.py" --feature "mean_5x5x5" --mp  --num_workers 3 --labels "L3 + L3R" "L3 + L3R_5x5x5"


# cercare total ROC features :
python "$DIR_PROJECT/total_roc/cercare_total_roc.py" --labels "L3 + L3R" "L3 + L3R_5x5x5"

# MRI total ROC features :
python "$DIR_PROJECT/total_roc/mri_total_roc.py" --labels "L3 + L3R" "L3 + L3R_5x5x5"

# Deactivate the conda environment
conda deactivate
