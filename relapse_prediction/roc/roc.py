from relapse_prediction import constants

from sklearn.metrics import roc_curve, auc
import pickle


def create_roc(df_data, label, feature_col, path_thresholds, path_roc_plot, title):

    fpr, tpr, thresholds = roc_curve(df_data[label], df_data[feature_col])
    d_res = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    # Save the thresholds :
    path_thresholds.parent.mkdir(parents=True, exist_ok=True)

    with open(path_thresholds, "wb") as f:
        pickle.dump(d_res, f)

    # AUC value :
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve :
    path_roc_plot.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    plt.savefig(str(path_roc_plot), dpi=300)
    plt.clf()
