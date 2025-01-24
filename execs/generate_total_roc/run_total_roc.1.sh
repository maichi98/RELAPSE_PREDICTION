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
VOXEL_STRATEGIES=("all_voxels")
PATIENT_STRATEGIES=("Class" "surgery_type" "all")

# Loop through each combination of voxel strategies and patient strategies
for voxel_strategy in "${VOXEL_STRATEGIES[@]}"; do
    for patient_strategy in "${PATIENT_STRATEGIES[@]}"; do
        python "$DIR_PROJECT/relapse_prediction/total_roc/cercare_total_roc.py" \
            --cercare_maps "CTH" "OEF" "rCBV"  \
            --labels "L3R" "L2"  "L3R - (L1 + L3)" "L2 + L3R" "L2 + L3R - (L1 + L3)"\
                     "L5" "L3" "L3 + L3R" "L1" "SumPreRT + L3R"\
            --voxel_strategy "$voxel_strategy" \
            --patient_strategy "$patient_strategy"


        # Run total ROC generation for MRI maps
        python "$DIR_PROJECT/relapse_prediction/total_roc/mri_total_roc.py" \
            --mri_maps "T1CE" "T1" "FLAIR" \
            --labels "L3R" "L2"  "L3R - (L1 + L3)" "L2 + L3R" "L2 + L3R - (L1 + L3)"\
                     "L5" "L3" "L3 + L3R" "L1" "SumPreRT + L3R"\
            --norms "z_score" \
            --voxel_strategy "$voxel_strategy" \
            --patient_strategy "$patient_strategy"

    done
done


# Deactivate the conda environment
conda deactivate