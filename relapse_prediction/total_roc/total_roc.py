from relapse_prediction import constants

from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle


def get_list_thresholds(label, feature_col):

    list_thresholds = set()

    for patient in constants.list_patients:

        path_thresholds = constants.dir_thresholds / patient / label / f"{feature_col}.pickle"
        with open(path_thresholds, "rb") as f:
            dict_thresholds = pickle.load(f)
        list_thresholds = list_thresholds.union(set(dict_thresholds["thresholds"]))

    return sorted(list(list_thresholds))


def _load_threshold_dataframe(patient, label, feature_col):

    path_thresholds = constants.dir_thresholds / patient / label / f"{feature_col}.pickle"
    with open(path_thresholds, "rb") as f:
        dict_thresholds = pickle.load(f)

    path_labels = constants.dir_labels / f"{patient}_labels.parquet"
    df_labels = pd.read_parquet(path_labels, engine="pyarrow")
    total_positives = len(df_labels[df_labels[label] == 1])
    total_negatives = len(df_labels[df_labels[label] == 0])

    df_thresholds = pd.DataFrame(columns=["thresholds", "TP", "FP"])

    df_thresholds["thresholds"] = dict_thresholds["thresholds"]

    df_thresholds["TP"] = dict_thresholds["tpr"] * total_positives
    df_thresholds["FP"] = dict_thresholds["fpr"] * total_negatives

    return df_thresholds, total_positives, total_negatives


def get_all_fpr_tpr(label, feature_col, list_thresholds):

    df_total = pd.DataFrame(columns=["thresholds", "TP", "FP"])
    df_total["thresholds"] = list(list_thresholds)
    df_total["TP"] = 0
    df_total["FP"] = 0
    total_positives = 0
    total_negatives = 0

    for patient in tqdm(constants.list_patients):

        df_thresholds = pd.DataFrame(list(list_thresholds), columns=["thresholds"])
        _df_thresholds, t_p, t_n = _load_threshold_dataframe(patient, label, feature_col)

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

    dir_save = constants.dir_results / "total_ROC" / label
    dir_save.mkdir(parents=True, exist_ok=True)

    with open(dir_save / f"{feature_col}_total_tpr_fpr.pickle", "wb") as f:
        pickle.dump(dict_fpr_tpr, f)

    return dict_fpr_tpr


def plot_total_roc(label, feature_col, dict_fpr_tpr):

    fpr = dict_fpr_tpr["fpr"]
    tpr = dict_fpr_tpr["tpr"]

    roc_auc = auc(fpr, tpr)

    dir_save = constants.dir_results / "total_ROC" / label
    dir_save.mkdir(parents=True, exist_ok=True)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve for label: {label}, feature: {feature_col}')
    plt.legend(loc="lower right")

    plt.savefig(str(dir_save / f"{feature_col}.png"), dpi=300)
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


def print_cutoff(label, feature_col, dict_fpr_tpr, file):

    tpr = np.array(dict_fpr_tpr["tpr"])
    fpr = np.array(dict_fpr_tpr["fpr"])
    thresholds = dict_fpr_tpr["thresholds"]

    roc_auc = auc(fpr, tpr)

    cutoff, recall, specificity = compute_cutoff_youden(fpr, tpr, thresholds, inverted=False)

    txt = f"Label={label:25s}, Feature={feature_col:35s}: Cutoff :  {cutoff}, Recall : {recall * 100:.2f}%"\
          f", Specificity : {specificity * 100:.2f}%, total AUC : {roc_auc:.2f}"

    file.write(txt + "\n")
