#!/bin/bash
# This script runs the ROC generation for all voxels for high priority setup 1.
# It separates the MRI and CERCARE runs using loops to reduce repetition.

# Exit immediately if a command fails
set -e

# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_relapse_prediction

# Define the project directory
DIR_PROJECT="/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION"

# Define arrays for the labels and the cercare maps
LABELS=("L3R" "L3R_5x5x5" "L2" "L2_5x5x5" "L3R - (L1 + L3)" "L3R - (L1 + L3)_5x5x5" \
        "L5" "L5_5x5x5" "L3" "L3_5x5x5" "L3 + L3R" "L3 + L3R_5x5x5" "L1" "L1_5x5x5" "L4" "L4_5x5x5")
CERCARE_MAPS=("CTH" "OEF" "rCBV" "rCMRO2" "COV" "rLeakage" "Delay")
FEATURES=("None" "mean_5x5x5")
REG_TPS=("Affine" "SyN")

VOXEL_STRATEGIES=("ALL_VOXELS")

##########################
# MRI ROC Calculations
##########################
echo "Starting MRI ROC calculations..."

# Loop over voxel strategies
for voxel_strategy in "${VOXEL_STRATEGIES[@]}"; do
    # Loop over registration types (Affine, SyN)
    for reg_tp in "${REG_TPS[@]}"; do
        # Loop over feature options ("None" and "mean_5x5x5")
        for feature in "${FEATURES[@]}"; do
            # Build the command as an array
            cmd=(python "$DIR_PROJECT/relapse_prediction/roc/mri_roc.py" \
                 --labels "${LABELS[@]}" \
                 --reg_tp "$reg_tp" \
                 --voxel_strategy "$voxel_strategy" \
                 --mp --num_workers 4 \
                 --feature "$feature")

            echo "Running MRI: reg_tp=$reg_tp, feature=$feature, voxel_strategy=$voxel_strategy"
            "${cmd[@]}"
            wait
        done
    done
done


##########################
# CERCARE ROC Calculations
##########################
echo "Starting CERCARE ROC calculations..."

# Loop over voxel strategies
for voxel_strategy in "${VOXEL_STRATEGIES[@]}"; do
    # Loop over registration types (Affine, SyN)
    for reg_tp in "Affine" "SyN"; do
        # Loop over feature options ("None" and "mean_5x5x5")
        for feature in "${FEATURES[@]}"; do
            # Build the command as an array
            cmd=(python "$DIR_PROJECT/relapse_prediction/roc/cercare_roc.py" \
                 --cercare_maps "${CERCARE_MAPS[@]}" \
                 --labels "${LABELS[@]}" \
                 --reg_tp "$reg_tp" \
                 --voxel_strategy "$voxel_strategy" \
                 --mp --num_workers 4 \
                 --feature "$feature")

            echo "Running CERCARE: reg_tp=$reg_tp, feature=$feature, voxel_strategy=$voxel_strategy"
            "${cmd[@]}"
            wait
        done
    done
done