#!/bin/bash

# This script runs the total ROC generation for top priority Cercare maps and top priority labels
# for all voxel strategies and all patient strategies

# Ensure that the script exits if a command fails
set -e

# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_relapse_prediction

# Define the project directory:
DIR_PROJECT="/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION"

# Define the voxel strategies and patient strategies
VOXEL_STRATEGIES=("all_voxels" "CTV" "OUTSIDE_CTV")
PATIENT_STRATEGIES=("all" "Class" "surgery_type" "IDH")

# Loop through each combination of voxel strategies and patient strategies
for voxel_strategy in "${VOXEL_STRATEGIES[@]}"; do
    for patient_strategy in "${PATIENT_STRATEGIES[@]}"; do
        python "$DIR_PROJECT/relapse_prediction/total_roc/cercare_total_roc.py" \
            --cercare_maps "Delay" "COV" "rLeakage" \
            --labels "L3R" "L3R_5x5x5" "L2" "L2_5x5x5" "L3R - (L1 + L3)" "L3R - (L1 + L3)_5x5x5"\
                     "L5" "L5_5x5x5" "L3" "L3_5x5x5" "L3 + L3R" "L3 + L3R_5x5x5" "L1" "L1_5x5x5" "L4" "L4_5x5x5"\
            --voxel_strategy "$voxel_strategy" \
            --patient_strategy "$patient_strategy"\
            --mp --num_workers 4
    done
done

# Deactivate the conda environment
conda deactivate