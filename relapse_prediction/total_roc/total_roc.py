from relapse_prediction import constants
from relapse_prediction import labels

from sklearn.metrics import auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle


def get_list_thresholds(list_patients, label, feature_col, dir_thresholds):

    list_thresholds = set()

    for patient in tqdm(list_patients):

        path_thresholds = (dir_thresholds
                           / patient
                           / fr"{label}_{feature_col}.pickle")

        if not path_thresholds.exists():
            print(fr"path_thresholds doesn't exist for patient {patient}, label {label},"
                  fr" feature {feature_col}, dir_thresholds {str(dir_thresholds)} !")

        else:
            with open(path_thresholds, "rb") as f:
                dict_thresholds = pickle.load(f)
            list_thresholds = list_thresholds.union(set(dict_thresholds["thresholds"]))

    return sorted(list(list_thresholds))


def _load_threshold_dataframe(patient, label, feature_col, dir_thresholds, reg_tp, voxel_strategy):

    voxel_strategy = voxel_strategy.upper()

    path_thresholds = (dir_thresholds
                       / patient
                       / f"{label}_{feature_col}.pickle")
    with open(path_thresholds, "rb") as f:
        dict_thresholds = pickle.load(f)

    df_labels = labels.get_df_labels(patient=patient, label=label, reg_tp=reg_tp)

    if voxel_strategy == "CERCARE_ONLY":
        df_labels = df_labels[df_labels["CERCARE"] == 1]
    elif voxel_strategy == "CERCARE_NO_VENTRICLES":
        df_labels = df_labels[(df_labels["CERCARE"] == 1) & (df_labels["Ventricles"] == 0)]

    total_positives = len(df_labels[df_labels[label] == 1])
    total_negatives = len(df_labels[df_labels[label] == 0])

    df_thresholds = pd.DataFrame(columns=["thresholds", "TP", "FP"])

    df_thresholds["thresholds"] = dict_thresholds["thresholds"]

    df_thresholds["TP"] = dict_thresholds["tpr"] * total_positives
    df_thresholds["FP"] = dict_thresholds["fpr"] * total_negatives

    return df_thresholds, total_positives, total_negatives


def get_all_fpr_tpr(list_patients, label, feature_col, dir_thresholds, list_thresholds,
                    reg_tp, voxel_strategy, path_total_thresholds):

    df_total = pd.DataFrame(columns=["thresholds", "TP", "FP"])
    df_total["thresholds"] = list(list_thresholds)
    df_total["TP"] = 0
    df_total["FP"] = 0
    total_positives = 0
    total_negatives = 0

    for patient in list_patients:

        df_thresholds = pd.DataFrame(list(list_thresholds), columns=["thresholds"])
        _df_thresholds, t_p, t_n = _load_threshold_dataframe(patient=patient,
                                                             label=label,
                                                             feature_col=feature_col,
                                                             dir_thresholds=dir_thresholds,
                                                             reg_tp=reg_tp,
                                                             voxel_strategy=voxel_strategy)

        if t_p > 0 and t_n > 0:
            df_thresholds = df_thresholds.merge(_df_thresholds, on="thresholds", how="left")

            df_thresholds = df_thresholds.sort_values(by="thresholds", ascending=True).reset_index(drop=True)

            df_thresholds["TP"] = df_thresholds["TP"].bfill()
            df_thresholds["FP"] = df_thresholds["FP"].bfill()

            idx = (df_thresholds["TP"].isna()) & (df_thresholds["FP"].isna())
            df_thresholds.loc[idx, "TP"] = 0
            df_thresholds.loc[idx, "FP"] = 0

            df_total["TP"] = df_total["TP"] + df_thresholds["TP"]
            df_total["FP"] = df_total["FP"] + df_thresholds["FP"]
            total_positives += t_p
            total_negatives += t_n

    df_total["tpr"] = df_total["TP"] / total_positives
    df_total["fpr"] = df_total["FP"] / total_negatives

    dict_fpr_tpr = {
        "fpr": list(df_total["fpr"]),
        "tpr": list(df_total["tpr"]),
        "thresholds": list(df_total["thresholds"])
    }

    path_total_thresholds.parent.mkdir(parents=True, exist_ok=True)

    with open(path_total_thresholds, "wb") as f:
        pickle.dump(dict_fpr_tpr, f)

    return dict_fpr_tpr


def plot_total_roc(dict_fpr_tpr, path_total_roc_plot, title):

    fpr = dict_fpr_tpr["fpr"]
    tpr = dict_fpr_tpr["tpr"]

    roc_auc = auc(fpr, tpr)

    path_total_roc_plot.parent.mkdir(parents=True, exist_ok=True)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    plt.savefig(str(path_total_roc_plot), dpi=300)
    plt.clf()


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

    # corresponding threshold :
    cutoff = thresholds[idx_max]

    # Recall and specificity :
    recall = tpr[idx_max]
    specificity = 1 - fpr[idx_max]

    # Return the corresponding threshold :
    return cutoff, recall, specificity


def add_cutoff(dict_fpr_tpr, df_data, dict_data):

    tpr = np.array(dict_fpr_tpr["tpr"])
    fpr = np.array(dict_fpr_tpr["fpr"])
    thresholds = dict_fpr_tpr["thresholds"]

    roc_auc = auc(fpr, tpr)

    cutoff, recall, specificity = compute_cutoff_youden(fpr, tpr, thresholds, inverted=False)

    dict_data["Cutoff"] = cutoff
    dict_data["Recall"] = recall
    dict_data["Specificity"] = specificity
    dict_data["total AUC"] = roc_auc

    df_data.loc[len(df_data)] = dict_data
