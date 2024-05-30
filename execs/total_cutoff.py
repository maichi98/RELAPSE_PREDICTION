from relapse_prediction import constants

import numpy as np
import pickle


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
        youden = fpr - tpr
    else:
        youden = tpr - fpr
    # Find the index of the point that maximizes the Youden index :
    idx_max = np.argmax(youden)
    # Return the corresponding threshold :
    return thresholds[idx_max]


if __name__ == "__main__":

    label = "L3R"
    imaging = "CTH"
    feature = "mean_5x5x5"

    with open(constants.dir_results / "total_ROC" / f"{label}_{imaging}_{feature}_total_tpr_fpr.pickle", "rb") as f:
        dict_total_thresholds = pickle.load(f)

    tpr = np.array(dict_total_thresholds["tpr"])
    fpr = np.array(dict_total_thresholds["fpr"])
    thresholds = dict_total_thresholds["thresholds"]

    cutoff = compute_cutoff_youden(fpr, tpr, thresholds, inverted=False)
    print(f"Cutoff value for label={label}, imaging={imaging}, feature={feature}", cutoff)

    cutoff = compute_cutoff_distance(fpr, tpr, thresholds, inverted=False)
    print(f"Cutoff distance value for label={label}, imaging={imaging}, feature={feature}", cutoff)
