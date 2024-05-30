from relapse_prediction import constants, features, labels

from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import ttest_ind
import seaborn as sn
import pickle
import os


def process_patient_label_imaging_feature(patient, label, imaging, feature):

    df_labels = labels.get_df_labels(patient)
    df_labels.rename(columns={f"mean_{label}_5x5x5": f"{label}_5x5x5"}, inplace=True)
    df_labels.loc[df_labels[f"{label}_5x5x5"] >= 0.5, f"{label}_5x5x5"] = 1
    df_labels.loc[df_labels[f"{label}_5x5x5"] < 0.5, f"{label}_5x5x5"] = 0

    df_features = features.get_df_imaging_features(patient, imaging)
    rename_columns = {col: f"{imaging}_{col}" for col in set(df_features.columns) - {'x', 'y', 'z', imaging}}
    df_features.rename(columns=rename_columns, inplace=True)

    df_data = df_labels.merge(df_features, on=["x", "y", "z"], how="left")

    import matplotlib.pyplot as plt

    ax = sn.boxplot(data=df_data, x=label, y=feature, showfliers=False, hue=label)
    dir_save = os.path.join(constants.dir_results, "boxplots per patient", label, imaging, feature)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    plt.savefig(os.path.join(str(dir_save), f"{patient}.png"), dpi=300)
    plt.clf()

    dir_save = os.path.join(constants.dir_results, "ttest_ind", label, imaging, feature)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    stat, p_value = ttest_ind(df_data[feature].loc[df_data[label] == 0],
                              df_data[feature].loc[df_data[label] == 1])

    d_ttest_ind = {"patient": patient,
                   "label": label,
                   "feature": feature,
                   "statistic": stat,
                   "p-value": p_value}

    with open(os.path.join(str(dir_save), f"{patient}.pkl"), "wb") as f:
        pickle.dump(d_ttest_ind, f)
    print(f"Patient {patient}, label {label}, imaging {imaging}, feature {feature} has been processed.")


def process_patient_label_imaging(patient, label, imaging):
    L_features = [imaging, f"{imaging}_mean_3x3", f"{imaging}_mean_5x5", f"{imaging}_mean_3x3x3", f"{imaging}_mean_5x5x5"]
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_patient_label_imaging_feature, patient, label, imaging, feature) for feature in L_features}
        for future in as_completed(futures):
            future.result()


def process_patient_label(patient, label):
    L_imaging = constants.L_CERCARE_MAPS + constants.L_IRM_MAPS
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_patient_label_imaging, patient, label, imaging) for imaging in L_imaging}
        for future in as_completed(futures):
            future.result()


def process_patient(patient):
    labels = ["L3R", "L3R - (L1 + L3)"]
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_patient_label, patient, label) for label in labels}
        for future in as_completed(futures):
            future.result()


def main():
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_patient, patient): patient for patient in constants.list_patients}
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()

