#!/bin/bash

# Ensure that the script exits if a command fails
set -e

# Activate the conda environment
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate env_relapse_prediction

# Define the project directory:
DIR_PROJECT="/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION/relapse_prediction"

# Run Features generation for priority maps :
python "$DIR_PROJECT/features/features.py" --mri_maps 'T1' 'T1CE' 'FLAIR' --cercare_maps "CTH" "OEF" "rCBV" --mp --num_workers 4

## Run Features generation for non priority maps :
python "$DIR_PROJECT/features/features.py" --cercare_maps 'COV' 'Delay' 'rLeakage' 'rCMRO2' --mp --num_workers 4

# Deactivate the conda environment
conda deactivate
