from relapse_prediction.total_roc.total_roc import (get_list_thresholds,
                                                    get_all_fpr_tpr,
                                                    plot_total_roc,
                                                    add_cutoff
                                                    )
from relapse_prediction import constants, utils

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import pandas as pd
import argparse
import pickle


def create_cercare_total_roc(imaging, label, reg_tp, feature, interpolator, voxel_strategy, patient_strategy, df_data):

    feature_col = f"{imaging}_{feature}" if feature is not None else imaging
    feature_col = f"{feature_col}_quantized"

    dict_list_patients = utils.get_list_patients_by_strategy(patient_strategy)

    for category, list_patients in dict_list_patients.items():

        path_thresholds = constants.dir_total_thresholds / voxel_strategy / label / category
        path_thresholds = path_thresholds / f"{feature_col}_total_tpr_fpr.pickle"

        if path_thresholds.exists():

            with open(path_thresholds, "rb") as f:
                dict_fpr_tpr = pickle.load(f)

            add_cutoff(label=label, feature_col=feature_col, dict_fpr_tpr=dict_fpr_tpr, df_data=df_data,
                       patient_category=category, voxel_strategy=voxel_strategy)

            print(f"Total ROC already exists! for imaging {imaging}, label {label}, feature {feature}, "
                  f"voxel strategy {voxel_strategy}, patient strategy {patient_strategy}")

        else:

            try:
                list_thresholds = get_list_thresholds(label=label, feature_col=feature_col,
                                                      voxel_strategy=voxel_strategy, list_patients=list_patients)

                dict_fpr_tpr = get_all_fpr_tpr(label=label, feature_col=feature_col, list_thresholds=list_thresholds,
                                               list_patients=list_patients, patient_category=category,
                                               voxel_strategy=voxel_strategy)

                plot_total_roc(label=label, feature_col=feature_col, dict_fpr_tpr=dict_fpr_tpr, patient_category=category,
                               voxel_strategy=voxel_strategy)

                add_cutoff(label=label, feature_col=feature_col, dict_fpr_tpr=dict_fpr_tpr, df_data=df_data,
                           patient_category=category, voxel_strategy=voxel_strategy)

            except Exception as e:
                print(f"Error in creating total ROC for imaging {imaging}, label {label}, feature {feature},"
                      f" voxel strategy {voxel_strategy}, patient strategy {patient_strategy}!")
                print(e)


def main(list_cercare_maps, list_labels, list_features, voxel_strategy, patient_strategy):

    dir_save = constants.dir_total_cutoffs / voxel_strategy / patient_strategy
    dir_save.mkdir(parents=True, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_data = dir_save / fr"Cercare_results_{current_time}.csv"

    df_data = pd.DataFrame(columns=["label", "Feature", "voxel strategy", "patient_category",
                                    "Cutoff", "Recall", "Specificity", "total AUC"])

    for imaging in list_cercare_maps:
        for label in list_labels:
            for feature in list_features:

                create_cercare_total_roc(imaging=imaging, label=label, feature=feature,
                                         voxel_strategy=voxel_strategy, patient_strategy=patient_strategy,
                                         df_data=df_data,
                )
                # print(f"Total ROC generated for imaging {imaging}, label {label}, feature {feature},"
                #       f" voxel strategy {voxel_strategy} patient strategy {patient_strategy}!")

                df_data.to_csv(path_data, index=False)


def process_imaging_label(tpl):
    imaging, label, feature, voxel_strategy, patient_strategy = tpl
    create_cercare_total_roc(imaging, label, feature, voxel_strategy, patient_strategy)
    print(f"Total ROC generated for imaging {imaging}, label {label}, feature {feature},"
          f" voxel strategy {voxel_strategy} patient strategy {patient_strategy}!")


def main_mp(list_mri_maps, list_labels, list_features, voxel_strategy, patient_strategy, num_workers):

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pairs = [(imaging, label, feature, voxel_strategy, patient_strategy)
                 for imaging in list_mri_maps
                 for label in list_labels
                 for feature in list_features]
        executor.map(process_imaging_label, pairs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create Total ROC curve for Cercare features')

    parser.add_argument('--cercare_maps', nargs='+', default=constants.L_CERCARE_MAPS,
                        help='list of Cercare images')

    parser.add_argument('--labels', nargs='+',
                        default=["L3R", "L3R_5x5x5", "L3R - (L1 + L3)", "L3R - (L1 + L3)_5x5x5",
                                 "L1", "L1_5x5x5", "L2", "L2_5x5x5", "L3", "L3_5x5x5",
                                 "L4", "L4_5x5x5", "L5", "L5_5x5x5"], help='list of Labels')

    parser.add_argument('--features', nargs='+', default=[None, "mean_5x5x5"],
                        help='list of Features')

    parser.add_argument('--voxel_strategy', default="all_voxels",
                        help='Voxel strategy, can be either all_voxels, CTV or OUTSIDE_CTV')

    parser.add_argument('--patient_strategy', default="all",
                        help='Patient strategy, can be either all or Class or surgery_type or IDH')

    parser.add_argument('--mp', action='store_true', default=False,
                        help='Use multiprocessing')

    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of workers for multiprocessing')

    args = parser.parse_args()

    features = [None if feature == 'None' else feature for feature in args.features]

    if args.mp:
        main_mp(list_mri_maps=args.cercare_maps,
                list_labels=args.labels,
                list_features=features,
                voxel_strategy=args.voxel_strategy,
                patient_strategy=args.patient_strategy,
                num_workers=args.num_workers)
    else:
        main(list_cercare_maps=args.cercare_maps,
             list_labels=args.labels,
             list_features=features,
             voxel_strategy=args.voxel_strategy,
             patient_strategy=args.patient_strategy)
