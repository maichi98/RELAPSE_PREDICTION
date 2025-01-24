from relapse_prediction import constants

from sklearn.metrics import roc_curve, auc
import pickle


def create_roc(df_data, patient, label, feature_col, voxel_strategy):

    if voxel_strategy not in ["all_voxels", "CTV", "OUTSIDE_CTV"]:
        raise ValueError(f"Invalid voxel strategy: {voxel_strategy}")

    fpr, tpr, thresholds = roc_curve(df_data[label], df_data[feature_col])
    d_res = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    path_thresholds = constants.dir_thresholds / voxel_strategy / patient / label / f"{feature_col}.pickle"
    path_thresholds.parent.mkdir(parents=True, exist_ok=True)

    with open(path_thresholds, "wb") as f:
        pickle.dump(d_res, f)

    # AUC value :
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve :
    dir_roc_plot = constants.dir_roc / voxel_strategy / label / feature_col
    dir_roc_plot.mkdir(exist_ok=True, parents=True)

    import matplotlib.pyplot as plt

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve for {patient}, label: {label}, feature: {feature_col}')
    plt.legend(loc="lower right")

    plt.savefig(str(dir_roc_plot / f"{patient}.png"), dpi=300)
    plt.clf()
