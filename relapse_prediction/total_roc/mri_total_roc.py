from relapse_prediction.total_roc.total_roc import (get_list_thresholds,
                                                    get_all_fpr_tpr,
                                                    plot_total_roc,
                                                    print_cutoff
                                                    )
from relapse_prediction import constants

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import argparse
import time
import os


def create_mri_total_roc(imaging, label, feature, norm, file):

    feature_col = f"{imaging}_{feature}" if feature is not None else imaging
    feature_col = f"{feature_col}_{norm}_normalized"

    list_thresholds = get_list_thresholds(label, feature_col)
    dict_fpr_tpr = get_all_fpr_tpr(label, feature_col, list_thresholds)
    plot_total_roc(label, feature_col, dict_fpr_tpr)
    print_cutoff(label, feature_col, dict_fpr_tpr, file)


def main(list_mri_maps, list_labels, list_features, list_norms):

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    file = open(constants.dir_results / "total_roc" / fr"IRM_results_{current_time}", "w")

    for imaging in list_mri_maps:
        for label in list_labels:
            for feature in list_features:
                for norm in list_norms:

                    create_mri_total_roc(imaging, label, feature, norm, file)
                    print(f"Total ROC generated for imaging {imaging}, label {label}, feature {feature}, norm {norm} !")

    file.close()


def process_imaging_label(tpl):
    imaging, label, feature, norm, file = tpl
    create_mri_total_roc(imaging, label, feature, norm, file)
    print(f"Total ROC generated for imaging {imaging}, label {label}, feature {feature}, norm {norm} !")


def main_mp(list_mri_maps, list_labels, list_features, list_norms,  num_workers):

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file = open(constants.dir_results / "total_roc" / fr"IRM_results_{current_time}", "w")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pairs = [(imaging, label, feature, norm, file)
                 for imaging in list_mri_maps
                 for label in list_labels
                 for feature in list_features
                 for norm in list_norms]
        executor.map(process_patient_imaging_label, pairs)

    file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create Total ROC curve for MRI features')

    parser.add_argument('--mri_maps', nargs='+', default=constants.L_IRM_MAPS,
                        help='list of MRI images')
    parser.add_argument('--labels', nargs='+',
                        default=["L3R", "L3R_5x5x5", "L3R - (L1 + L3)", "L3R - (L1 + L3)_5x5x5"], help='list of Labels')
    parser.add_argument('--features', default=[None, 'mean_5x5x5'], nargs='+',
                        help="choice of feature")
    parser.add_argument('--norms', type=str, default=['z_score', 'min_max', 'max'],
                        help='normalization method of the features')
    parser.add_argument('--mp', action='store_true', default=False,
                        help='Use multiprocessing ?')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help='number of CPU workers')

    args = parser.parse_args()

    if not args.mp:
        main(list_mri_maps=args.mri_maps,
             list_labels=args.labels,
             list_features=args.features,
             list_norms=args.norms)
    else:
        main_mp(list_mri_maps=args.mri_maps,
                list_labels=args.labels,
                list_features=args.features,
                list_norms=args.norms,
                num_workers=args.num_workers)
