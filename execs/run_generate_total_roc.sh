#!/bin/bash

# Ensure that the script exits if a command fails
set -e

# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_relapse_prediction

# Define the project directory:
DIR_PROJECT="/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION"

source "$DIR_PROJECT/execs/generate_total_roc/run_total_roc.1.sh"

source "$DIR_PROJECT/execs/generate_total_roc/run_total_roc.2.sh"


conda deactivate
