#!/bin/bash

# Ensure that the script exits if a command fails
set -e

# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_relapse_prediction

# Define the project directory:
DIR_PROJECT="/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION"


# Run priority setup 1 for the three voxel strategies:
source "$DIR_PROJECT/execs/generate_roc_for_all_voxels/run_all_voxels.1.sh"
#source "$DIR_PROJECT/execs/generate_roc_for_cercare_only_voxels/run_cercare_only_voxels.1.sh"
#source "$DIR_PROJECT/execs/generate_roc_for_cercare_no_ventricles_voxels/run_cercare_no_ventricles_voxels.1.sh"

# Run priority setup 2 for the three voxel strategies:
source "$DIR_PROJECT/execs/generate_roc_for_all_voxels/run_all_voxels.2.sh"
#source "$DIR_PROJECT/execs/generate_roc_for_cercare_only_voxels/run_cercare_only_voxels.2.sh"
#source "$DIR_PROJECT/execs/generate_roc_for_cercare_no_ventricles_voxels/run_cercare_no_ventricles_voxels.2.sh"

# Run priority setup 3 for the three voxel strategies:
source "$DIR_PROJECT/execs/generate_roc_for_all_voxels/run_all_voxels.3.sh"
#source "$DIR_PROJECT/execs/generate_roc_for_cercare_only_voxels/run_cercare_only_voxels.3.sh"
#source "$DIR_PROJECT/execs/generate_roc_for_cercare_no_ventricles_voxels/run_cercare_no_ventricles_voxels.2.sh"

conda deactivate
