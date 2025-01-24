#!/bin/bash

# This script runs the ROC generation for all voxels for lesser priority setup 4

# Ensure that the script exits if a command fails
set -e

# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_relapse_prediction

# Define the project directory:
DIR_PROJECT="/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION"

# ROC for low priority cercare features :
python "$DIR_PROJECT/relapse_prediction/roc/cercare_roc.py" \
        --cercare_maps "COV" "Delay" "rLeakage" \
        --labels "L3R" "L3R_5x5x5" "L2" "L2_5x5x5" "L3R - (L1 + L3)" "L3R - (L1 + L3)_5x5x5"\
                  "L5" "L5_5x5x5" "L3" "L3_5x5x5" "L3 + L3R" "L3 + L3R_5x5x5" "L1" "L1_5x5x5" "L4" "L4_5x5x5" \
        --voxel_strategy "all_voxels"\
        --mp  --num_workers 10

wait

# ROC for low priority cercare features  feature_5x5x5:
python "$DIR_PROJECT/relapse_prediction/roc/cercare_roc.py" \
        --cercare_maps "COV" "Delay" "rLeakage" \
        --feature "mean_5x5x5" \
        --labels "L3R" "L3R_5x5x5" "L2" "L2_5x5x5" "L3R - (L1 + L3)" "L3R - (L1 + L3)_5x5x5"\
                  "L5" "L5_5x5x5" "L3" "L3_5x5x5" "L3 + L3R" "L3 + L3R_5x5x5" "L1" "L1_5x5x5" "L4" "L4_5x5x5" \
        --voxel_strategy "all_voxels"\
        --mp  --num_workers 10

wait

# Deactivate the conda environment
conda deactivate
