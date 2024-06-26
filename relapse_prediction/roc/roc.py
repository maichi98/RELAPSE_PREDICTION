from relapse_prediction import constants

from sklearn.metrics import roc_curve, auc
import pickle


def create_roc(df_data, patient, label, feature_col):

    fpr, tpr, thresholds = roc_curve(df_data[label], df_data[feature_col])
    d_res = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    path_roc_results = constants.dir_results / "thresholds per patient" / patient / label / f"{feature_col}.pickle"
    path_roc_results.parent.mkdir(parents=True, exist_ok=True)

    with open(path_roc_results, "wb") as f:
        pickle.dump(d_res, f)

    # AUC value :
    roc_auc = auc(fpr, tpr)
    dir_save = constants.dir_results / "ROC per patient" / label / feature_col
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
