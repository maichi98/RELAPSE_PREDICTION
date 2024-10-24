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
    "list_patients",
    "D_KERNELS",
    "dict_cercare_p"
]

# Hard drive directory :
dir_root = Path("/media/maichi/T7") if platform.system() == "Linux" else Path("D:")

# AIDREAM_DATA directory :
dir_aidream_data = dir_root / "PERFUSION_DATA"

# processed imaging directory :
dir_processed = dir_aidream_data / "PROCESSED"

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
list_bad_patients = {"AIDREAM_133", "AIDREAM_201", 'AIDREAM_102,'}
list_patients = list(set(os.listdir(dir_processed)) - list_bad_patients)
list_patients = [list_patients[i] for i in np.argsort([int(patient.strip("AIDREAM_")) for patient in list_patients])]


D_KERNELS = {
     "mean_5x5x5": np.ones((5, 5, 5)) / 125
}

dict_cercare_p = {
    "COV": 2, "CTH": 2, "Delay": 2, "rCBV": 2,
    "rLeakage": 1, "OEF": 3, "rCMRO2": 2
}
