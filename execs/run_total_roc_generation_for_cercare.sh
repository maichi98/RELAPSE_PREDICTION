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

# Define labels for MRI maps ROC generation
LABELS=(
    "L3R" "L3R_5x5x5"
    "L2" "L2_5x5x5"
    "L3R - (L1 + L3)" "L3R - (L1 + L3)_5x5x5"
    "L5" "L5_5x5x5"
    "L3" "L3_5x5x5"
    "L3 + L3R" "L3 + L3R_5x5x5"
    "L1" "L1_5x5x5"
    "L4" "L4_5x5x5"
)
CERCARE_MAPS=("CTH" "OEF" "rCBV" "rCMRO2" "COV" "rLeakage" "Delay")

# Define voxel strategies
VOXEL_STRATEGIES=("ALL_VOXELS")

#####################################
# Run total ROC generation for "Affine" registration with all patient strategies
#####################################
echo "Starting total ROC generation using 'Affine' registration..."

PATIENT_STRATEGIES_AFFINE=("all" "SyN_patients" "Class" "surgery_type")
for voxel_strategy in "${VOXEL_STRATEGIES[@]}"; do
    for patient_strategy in "${PATIENT_STRATEGIES_AFFINE[@]}"; do
        echo "Processing Affine registration | Voxel Strategy: ${voxel_strategy} | Patient Strategy: ${patient_strategy}"
        python "$DIR_PROJECT/relapse_prediction/total_roc/cercare_total_roc.py" \
            --cercare_maps "${CERCARE_MAPS[@]}" \
            --labels "${LABELS[@]}" \
            --reg_tps "Affine" \
            --voxel_strategy "$voxel_strategy" \
            --patient_strategy "$patient_strategy"
    done
done

#####################################
# Run total ROC generation for "SyN" registration with a subset of patient strategies
#####################################
echo "Starting total ROC generation using 'SyN' registration..."

PATIENT_STRATEGIES_SYN=("SyN_patients" "Class" "surgery_type")
for voxel_strategy in "${VOXEL_STRATEGIES[@]}"; do
    for patient_strategy in "${PATIENT_STRATEGIES_SYN[@]}"; do
        echo "Processing SyN registration | Voxel Strategy: ${voxel_strategy} | Patient Strategy: ${patient_strategy}"
        python "$DIR_PROJECT/relapse_prediction/total_roc/cercare_total_roc.py" \
            --cercare_maps "${CERCARE_MAPS[@]}" \
            --labels "${LABELS[@]}" \
            --reg_tps "SyN" \
            --voxel_strategy "$voxel_strategy" \
            --patient_strategy "$patient_strategy"
    done
done

# Deactivate the conda environment
conda deactivate