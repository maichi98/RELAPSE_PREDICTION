from relapse_prediction import constants

import pickle


def get_list_mri_thresholds(imaging, label, feature, norm):

    feature_col = f"{imaging}_{feature}" if feature is not None else imaging
    feature_col = f"{feature}_{norm}_normalized"

    list_thresholds = set()

    for patient in constants.list_patients:

        path_thresholds = constants.dir_thresholds / patient / f"{feature_col}.pickle"
        with open(path_thresholds, "rb") as f:
            dict_thresholds = pickle.load(f)
        list_thresholds = list_thresholds.union(set(dict_thresholds["thresholds"]))

    return sorted(list(list_thresholds))


def load_mri_threshold_dataframe(patient, imaging, label, feature, norm):

    feature_col = f"{imaging}_{feature}" if feature is not None else imaging
    feature_col = f"{feature}_{norm}_normalized"

    path_thresholds = constants.dir_thresholds / patient / f"{feature_col}.pickle"
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