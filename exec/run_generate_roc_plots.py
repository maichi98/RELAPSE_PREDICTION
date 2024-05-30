from relapse_prediction import constants, features, labels

from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import roc_curve, auc
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

    fpr, tpr, thresholds = roc_curve(df_data[f"{label}_5x5x5"], df_data[feature])
    d_res = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }

    path_roc_results = constants.dir_results / "thresholds per patient" / f"{label}_5x5x5" / imaging / feature / f"{patient}.pickle"
    path_roc_results.parent.mkdir(parents=True, exist_ok=True)

    with open(path_roc_results, "wb") as f:
        pickle.dump(d_res, f)

    # AUC value :
    roc_auc = auc(fpr, tpr)
    dir_save = os.path.join(constants.dir_results, "ROC per patient", f"{label}_5x5x5", imaging, feature)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    import matplotlib.pyplot as plt

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve for {patient}, label: {f"{label}_5x5x5"}, imaging: {imaging}')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(str(dir_save), f"{patient}.png"), dpi=300)
    plt.clf()

    d_auc = {"patient": patient,
             "label": f"{label}_5x5x5",
             "feature": feature,
             "AUC": roc_auc}

    dir_save = os.path.join(constants.dir_results, "AUC", f"{label}_5x5x5", imaging, feature)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    with open(os.path.join(str(dir_save), f"{patient}.pkl"), "wb") as f:
        pickle.dump(d_auc, f)

    print(f"Patient {patient}, label {f'{label}_5x5x5'}, imaging {imaging}, feature {feature} has been processed.")


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
    # labels = ["(L1 + L3)", "L2", "L3", "L4", "L5", "(L4 + L5)", "L3R", "L3R - (L1 + L3)"]
    labels = ["(L1 + L3)", "L2", "L3", "L4", "L5", "(L4 + L5)", "L3R", "L3R - (L1 + L3)"]
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_patient_label, patient, label) for label in labels}
        for future in as_completed(futures):
            future.result()


def main():
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_patient, patient): patient for patient in constants.list_patients}
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()



