from relapse_prediction import constants, features, labels

from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import seaborn as sn
import pandas as pd
import pickle
import os


def process_patient_label_imaging_feature(patient, label, imaging, feature):

    path_labels = constants.dir_labels / f"{patient}_labels.parquet"
    df_labels = pd.read_parquet(path_labels, engine="pyarrow")
    df_labels = df_labels[["x", "y", "z", label]]

    path_features = constants.dir_features / patient / fr"{patient}_{imaging}_features.parquet"
    df_features = pd.read_parquet(path_features, engine="pyarrow")
    rename_columns = {col: f"{imaging}_{col}" for col in set(df_features.columns) - {'x', 'y', 'z', imaging}}
    df_features.rename(columns=rename_columns, inplace=True)
    feature = f"{imaging}_{feature}" if imaging != feature else feature
    df_features = df_features[['x', 'y', 'z', feature]]

    df_data = df_labels.merge(df_features, on=["x", "y", "z"], how="left")
    
    fpr, tpr, thresholds = roc_curve(df_data[label], df_data[feature])
    d_res = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }
    
    path_roc_results = constants.dir_results / "thresholds per patient" / label / imaging / feature / f"{patient}.pickle"
    path_roc_results.parent.mkdir(parents=True, exist_ok=True)

    with open(path_roc_results, "wb") as f:
        pickle.dump(d_res, f)

    # AUC value :
    roc_auc = auc(fpr, tpr)
    dir_save = constants.dir_results / "ROC per patient" / label / imaging / feature
    dir_save.mkdir(exist_ok=True, parents=True)

    import matplotlib.pyplot as plt

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve for {patient}, label: {label}, imaging: {imaging}')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(str(dir_save), f"{patient}.png"), dpi=300)
    plt.clf()

    print(f"Patient {patient}, label {label}, imaging {imaging}, feature {feature} has been processed.")


def main():

    for patient in tqdm(constants.list_patients):
        for label in ["L3R", "L3R_5x5x5", "L3R - (L1 + L3)", "L3R - (L1 + L3)_5x5x5"]:
            for imaging in constants.L_IRM_MAPS:
                for feature in [imaging, "mean_5x5x5"]:
                    process_patient_label_imaging_feature(patient, label, imaging, feature)


if __name__ == "__main__":
    main()
