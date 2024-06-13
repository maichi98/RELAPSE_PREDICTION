from pathlib import Path
import numpy as np
import platform
import os

__all__ = [
    "dir_root",
    "dir_aidream_data",
    "dir_processed",
    "dir_features",
    "dir_labels",
    "dir_results",
    "dir_thresholds",
    "L_CERCARE_MAPS",
    "L_IRM_MAPS",
    "list_bad_patients",
    "list_patients",
    "D_KERNELS"
]

# Hard drive directory :
dir_root = Path("/media/maichi/SSD-IGR") if platform.system() == "Linux" else Path("E:")

# AIDREAM_DATA directory :
dir_aidream_data = dir_root / "AIDREAM_DATA"

# processed imaging directory :
dir_processed = dir_aidream_data / "processed"

# features directory :
dir_features = dir_aidream_data / "features"
dir_features.mkdir(exist_ok=True)

# Labels directory :
dir_labels = dir_aidream_data / "labels"
dir_labels.mkdir(exist_ok=True)

# Results directory :
dir_results = dir_root / "Relapse prediction" / "results"
dir_results.mkdir(parents=True, exist_ok=True)

# thresholds per patient directory :
dir_thresholds = dir_results / "thresholds per patient"

# List of cercare maps :
L_CERCARE_MAPS = ["COV", "CTH", "Delay", "rCBV", "rLeakage", "OEF", "rCMRO2"]
L_IRM_MAPS = ["T1CE", "T1", "FLAIR"]

# List of AIDREAM patients :
list_bad_patients = {"AIDREAM_32", "AIDREAM_102", "AIDREAM_238"}
list_patients = list(set(os.listdir(dir_processed)) - list_bad_patients)
list_patients = [list_patients[i] for i in np.argsort([int(patient.strip("AIDREAM_")) for patient in list_patients])]

D_KERNELS = {
    "mean_3x3": np.ones((3, 3)) / 9,
    "mean_5x5": np.ones((5, 5)) / 25,
    "mean_3x3x3": np.ones((3, 3, 3)) / 27,
    "mean_5x5x5": np.ones((5, 5, 5)) / 125
}

# D_KERNELS = {
#     "mean_5x5x5": np.ones((5, 5, 5)) / 125
# }
