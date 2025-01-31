from pathlib import Path
import numpy as np
import platform
import os

DIR_HARD_DRIVE = Path("/media/maichi/T7") if platform.system() == "Linux" else Path("D:/")

DIR_PROCESSED = DIR_HARD_DRIVE / "PERFUSION_DATA" / "PROCESSED"
DIR_FEATURES = DIR_HARD_DRIVE / "PERFUSION_DATA" / "FEATURES"
DIR_LABELS = DIR_HARD_DRIVE / "PERFUSION_DATA" / "LABELS"
DIR_THRESHOLDS = DIR_HARD_DRIVE / "PERFUSION_DATA" / "THRESHOLDS"
DIR_ROC_PLOTS = DIR_HARD_DRIVE / "PERFUSION_DATA" / "ROC_PLOTS"
DIR_TOTAL_THRESHOLDS = DIR_HARD_DRIVE / "PERFUSION_DATA" / "TOTAL_THRESHOLDS"
DIR_TOTAL_ROC_PLOTS = DIR_HARD_DRIVE / "PERFUSION_DATA" / "TOTAL_ROC_PLOTS"

LIST_MRI_MAPS = ["T1", "T1CE", "FLAIR"]
LIST_CERCARE_MAPS = ["CTH", "rCBV", "OEF", "rCMRO2", "Delay", "rLeakage", "COV"]
LIST_INTERPOLATORS = ["linear", "bSpline", "lanczosWindowedSinc", "nearestNeighbor", "genericLabel"]

DIR_DATA = Path(__file__).parent.parent / "data"

PATH_PERFUSION_PATIENTS = DIR_DATA / "perfusion_patients.txt"
PATH_SPLITTING_STRATEGY = DIR_DATA / "splitting_strategy.xlsx"
PATH_REFERENTIAL_TABLE = DIR_DATA / "referential_table.csv"
DICT_CERCARE_P = {
    "CTH": 2,
    "OEF": 3,
    "rCBV": 2,
    "rCMRO2": 2,
    "Delay": 2,
    "COV": 2,
    "rLeakage": 1
}

D_KERNELS = {
     "mean_5x5x5": np.ones((5, 5, 5)) / 125
}
