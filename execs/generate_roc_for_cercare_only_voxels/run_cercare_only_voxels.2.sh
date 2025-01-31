#!/bin/bash

# This script runs the ROC generation for all voxels for high priority setup 1

# Ensure that the script exits if a command fails
set -e

# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_relapse_prediction

# Define the project directory:
DIR_PROJECT="/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION"

# ROC for MRI features for high priority labels with Affine registration for labels :
python "$DIR_PROJECT/relapse_prediction/roc/mri_roc.py" \
        --labels "L5" "L5_5x5x5" "L3" "L3_5x5x5" "L3 + L3R" "L3 + L3R_5x5x5" "L1" "L1_5x5x5" "L4" "L4_5x5x5"\
        --reg_tp "Affine" \
        --voxel_strategy "CERCARE_ONLY"\
        --mp  --num_workers 3

wait

# ROC for high priority cercare features for high priority labels with Affine registration for labels :
python "$DIR_PROJECT/relapse_prediction/roc/cercare_roc.py" \
        --cercare_maps "CTH" "OEF" "rCBV" "rCMRO2" \
        --labels "L5" "L5_5x5x5" "L3" "L3_5x5x5" "L3 + L3R" "L3 + L3R_5x5x5" "L1" "L1_5x5x5" "L4" "L4_5x5x5"\
        --reg_tp "Affine" \
        --voxel_strategy "CERCARE_ONLY"\
        --mp  --num_workers 3
wait

# ROC for MRI features for high priority labels with SyN registration for labels :
python "$DIR_PROJECT/relapse_prediction/roc/mri_roc.py" \
        --labels "L5" "L5_5x5x5" "L3" "L3_5x5x5" "L3 + L3R" "L3 + L3R_5x5x5" "L1" "L1_5x5x5" "L4" "L4_5x5x5"\
        --reg_tp "SyN" \
        --voxel_strategy "CERCARE_ONLY"\
        --mp  --num_workers 3
wait

# ROC for high priority cercare features for high priority labels with SyN registration for labels :
python "$DIR_PROJECT/relapse_prediction/roc/cercare_roc.py" \
        --cercare_maps "CTH" "OEF" "rCBV" "rCMRO2" \
        --labels "L5" "L5_5x5x5" "L3" "L3_5x5x5" "L3 + L3R" "L3 + L3R_5x5x5" "L1" "L1_5x5x5" "L4" "L4_5x5x5"\
        --reg_tp "SyN" \
        --voxel_strategy "CERCARE_ONLY"\
        --mp  --num_workers 3
wait

# Deactivate the conda environment
conda deactivate
