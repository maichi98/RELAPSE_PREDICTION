#!/bin/bash

# This script runs the ROC generation for OUTSIDE_CTV voxels for high priority setup 1

# Ensure that the script exits if a command fails
set -e

# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_relapse_prediction

# Define the project directory:
DIR_PROJECT="/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION"

# ROC for MRI features for high priority labels :
python "$DIR_PROJECT/relapse_prediction/roc/mri_roc.py" \
        --labels "L3R" "L3R_5x5x5" "L2" "L2_5x5x5" "L3R - (L1 + L3)" "L3R - (L1 + L3)_5x5x5" \
        --voxel_strategy "OUTSIDE_CTV"\
        --mp  --num_workers 3


# ROC for high priority cercare features for high priority labels :
python "$DIR_PROJECT/relapse_prediction/roc/cercare_roc.py" \
        --cercare_maps "CTH" "OEF" "rCBV" "rCMRO2" \
        --labels "L3R" "L3R_5x5x5" "L2" "L2_5x5x5" "L3R - (L1 + L3)" "L3R - (L1 + L3)_5x5x5" \
        --voxel_strategy "OUTSIDE_CTV"\
        --mp  --num_workers 3

# Deactivate the conda environment
conda deactivate
