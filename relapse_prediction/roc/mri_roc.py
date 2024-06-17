from relapse_prediction import features, labels, constants

from sklearn.metrics import roc_curve, auc
import pickle


def create_mri_roc(patient, imaging, label, feature, norm):

    df_labels = labels.get_df_labels(patient, label)
    df_features = features.get_mri_features(patient, imaging, feature, norm)
    df_data = df_labels.merge(df_features, on=["x", "y", "z"], how="left")

    feature_col = f"{imaging}_{feature}" if feature is not None else imaging
    feature_col = f"{feature_col}_{norm}_normalized"

    fpr, tpr, thresholds = roc_curve(df_data[label], df_data[feature_col])
    d_res = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    path_roc_results = constants.dir_results / "thresholds per patient" / patient / f"{feature_col}.pickle"
    path_roc_results.parent.mkdir(parents=True, exist_ok=True)

    with open(path_roc_results, "wb") as f:
        pickle.dump(d_res, f)

    # AUC value :
    roc_auc = auc(fpr, tpr)
    dir_save = constants.dir_results / "ROC per patient" / label / imaging / feature_col
    dir_save.mkdir(exist_ok=True, parents=True)

    import matplotlib.pyplot as plt

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve for {patient}, label: {label}, feature: {feature_col}')
    plt.legend(loc="lower right")

    plt.savefig(str(dir_save / f"{patient}.png"), dpi=300)
    plt.clf()
