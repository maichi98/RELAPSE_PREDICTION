from relapse_prediction import constants
from pathlib import Path
from tqdm import tqdm
import shutil
import pandas as pd
import os
import pickle
from sklearn.metrics import roc_curve, auc
import numpy as np


def compute_cutoff_distance(fpr, tpr, thresholds, inverted=False):
    """
    Compute the cutoff value for the ROC curve.
    :param fpr: False Positive Rate
    :param tpr: True Positive Rate
    :param thresholds: Thresholds
    :return: Cutoff value
    """
    # Compute the distance to the point (0, 1) for each point of the ROC curve :
    if inverted:
        distances = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    else:
        distances = np.sqrt((1 - fpr) ** 2 + tpr ** 2)
    # Find the index of the point that minimizes the distance :
    idx_min = np.argmin(distances)
    # Return the corresponding threshold :
    return thresholds[idx_min]


def compute_cutoff_youden(fpr, tpr, thresholds, inverted=False):
    """
    Compute the cutoff value for the ROC curve.
    :param fpr: False Positive Rate
    :param tpr: True Positive Rate
    :param thresholds: Thresholds
    :return: Cutoff value
    """
    # Compute the Youden index for each point of the ROC curve :
    if inverted:
        youden = fpr + (1 - tpr) - 1
    else:
        youden = tpr + (1 - fpr) - 1
    # Find the index of the point that maximizes the Youden index :
    idx_max = np.argmax(youden)
    # Return the corresponding threshold :
    return thresholds[idx_max]


if __name__ == "__main__":

    # list_labels = ["(L1 + L3)", "L2", "L3", "L4", "L5", "(L4 + L5)", "L3R", "L3R - (L1 + L3)"]
    list_labels = ["L4", "L5", "(L4 + L5)"]
    list_labels += [f"{label}_5x5x5" for label in list_labels]
    list_imaging = constants.L_CERCARE_MAPS + constants.L_IRM_MAPS + ["olea rCBV"]

    df_results = pd.DataFrame(columns=["patient", "label", "imaging", "feature", "AUC", "cutoff-distance", "cutoff-youden"])

    for patient in tqdm(constants.list_patients):

        for label in list_labels:

            for imaging in list_imaging:

                list_features = [imaging, f"{imaging}_mean_3x3", f"{imaging}_mean_5x5", f"{imaging}_mean_3x3x3", f"{imaging}_mean_5x5x5"]

                for feature in list_features:

                    path_roc = constants.dir_results / "thresholds per patient" / label / imaging / feature / f"{patient}.pickle"

                    if not path_roc.exists():
                        df_results.loc[len(df_results)] = [patient, label, imaging, feature, None, None, None]
                    else:
                        with open(path_roc, "rb") as f:
                            d_roc = pickle.load(f)
                        fpr, tpr, thresholds = d_roc["fpr"], d_roc["tpr"], d_roc["thresholds"]

                        val_auc = auc(fpr, tpr)
                        val_cutoff_distance = compute_cutoff_distance(fpr, tpr, thresholds, inverted=True)
                        val_cutoff_youden = compute_cutoff_youden(fpr, tpr, thresholds, inverted=True)

                        df_results.loc[len(df_results)] = {"patient": patient,
                                                           "label": label,
                                                           "imaging": imaging,
                                                           "feature": feature,
                                                           "AUC": val_auc,
                                                           "cutoff-distance": val_cutoff_distance,
                                                           "cutoff-youden": val_cutoff_youden}

    df_results.to_csv(constants.dir_results / "auc_cutoff_results_inverted.csv", index=False)
