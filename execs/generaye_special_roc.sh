#!/bin/bash

# Ensure that the script exits if a command fails
set -e

# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_relapse_prediction

# Define the project directory:
DIR_PROJECT="/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION"

python "$DIR_PROJECT/relapse_prediction/roc/mri_roc.py" \
        --labels "L2 + L3R" "L2 + L3R - (L1 + L3)"  \
        --voxel_strategy "all_voxels"\
        #--overwrite &
        #--mp  --num_workers 2

wait

# ROC for high priority cercare features for high priority labels :
python "$DIR_PROJECT/relapse_prediction/roc/cercare_roc.py" \
        --cercare_maps "CTH" "OEF" "rCBV"  \
        --labels "L2 + L3R" "L2 + L3R - (L1 + L3)"  \
        --voxel_strategy "all_voxels"\
        --overwrite \
        --mp  --num_workers 2

wait

python "$DIR_PROJECT/relapse_prediction/roc/mri_roc.py" \
        --labels "L2 + L3R" "L2 + L3R - (L1 + L3)"  \
        --voxel_strategy "all_voxels"\
        --feature "mean_5x5x5"\
        --overwrite &
        #--mp  --num_workers 2\

wait

# ROC for high priority cercare features for high priority labels :
python "$DIR_PROJECT/relapse_prediction/roc/cercare_roc.py" \
        --cercare_maps "CTH" "OEF" "rCBV"  \
        --labels "L2 + L3R" "L2 + L3R - (L1 + L3)"  \
        --voxel_strategy "all_voxels"\
        --feature "mean_5x5x5" \
        --overwrite
        #--mp  --num_workers 2

wait

# Deactivate the conda environment
conda deactivate