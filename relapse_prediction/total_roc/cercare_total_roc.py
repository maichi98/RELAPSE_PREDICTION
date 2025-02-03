from relapse_prediction.total_roc.total_roc import (get_list_thresholds,
                                                    get_all_fpr_tpr,
                                                    plot_total_roc,
                                                    add_cutoff
                                                    )
from relapse_prediction import constants, utils

from datetime import datetime
import pandas as pd
import argparse
import pickle


def create_cercare_total_roc(imaging, label, reg_tp, feature, interpolator, voxel_strategy, patient_strategy, df_data):

    voxel_strategy = voxel_strategy.upper()

    if voxel_strategy not in ["ALL_VOXELS", "CERCARE_ONLY", "CERCARE_NO_VENTRICLES"]:
        raise ValueError("voxel_strategy must be either 'ALL_VOXELS', 'CERCARE_ONLY' or 'CERCARE_NO_VENTRICLES' !")

    feature_col = f"{imaging}_{feature}" if feature is not None else imaging
    feature_col = f"{feature_col}_quantized"

    dict_list_patients = utils.get_list_patients_by_strategy(patient_strategy, reg_tp)

    for category, list_patients in dict_list_patients.items():

        path_total_thresholds = (constants.DIR_TOTAL_THRESHOLDS
                                 / voxel_strategy
                                 / reg_tp
                                 / "CERCARE"
                                 / interpolator
                                 / category
                                 / f"{label}_{feature_col}_total_tpr_fpr.pickle")

        if path_total_thresholds.exists():

            with open(path_total_thresholds, "rb") as f:
                dict_fpr_tpr = pickle.load(f)

            dict_data = {"label": label, "Feature": feature_col, "reg_tp": reg_tp, "interpolator": interpolator,
                         "voxel strategy": voxel_strategy, "patient_category": category}

            add_cutoff(dict_fpr_tpr=dict_fpr_tpr,
                       df_data=df_data,
                       dict_data=dict_data)

            print(f"Total ROC already exists! for imaging {imaging}, label {label}, feature {feature}, "
                  f"voxel strategy {voxel_strategy}, patient strategy {patient_strategy}")

        else:

            try:
                dir_thresholds = (constants.DIR_THRESHOLDS
                                  / voxel_strategy
                                  / reg_tp
                                  / "CERCARE"
                                  / interpolator)
                if not dir_thresholds.exists():
                    raise ValueError(f"dir_thresholds doesn't exist for reg_tp {reg_tp}, voxel_strategy {voxel_strategy},"
                                     f" interpolator {interpolator}!")

                list_thresholds = get_list_thresholds(list_patients=list_patients,
                                                      label=label,
                                                      feature_col=feature_col,
                                                      dir_thresholds=dir_thresholds)

                dict_fpr_tpr = get_all_fpr_tpr(list_patients=list_patients,
                                               label=label,
                                               feature_col=feature_col,
                                               dir_thresholds=dir_thresholds,
                                               list_thresholds=list_thresholds,
                                               reg_tp=reg_tp,
                                               voxel_strategy=voxel_strategy,
                                               path_total_thresholds=path_total_thresholds)

                path_total_roc_plot = (constants.DIR_TOTAL_ROC_PLOTS
                                       / voxel_strategy
                                       / reg_tp
                                       / "CERCARE"
                                       / interpolator
                                       / category
                                       / f"{label}_{feature_col}_total_roc.png")

                title = fr"ROC curve for label : {label} ({reg_tp}) {feature_col} ({interpolator})"

                plot_total_roc(dict_fpr_tpr=dict_fpr_tpr,
                               path_total_roc_plot=path_total_roc_plot,
                               title=title)

                dict_data = {"label": label, "Feature": feature_col, "reg_tp": reg_tp, "interpolator": interpolator,
                             "voxel strategy": voxel_strategy, "patient_category": category}

                add_cutoff(dict_fpr_tpr=dict_fpr_tpr,
                           df_data=df_data,
                           dict_data=dict_data)

            except Exception as e:
                print(f"Error in creating total ROC for imaging {imaging}, label {label}, reg_tp {reg_tp}, feature {feature},"
                      f"interpolator {interpolator}  voxel strategy {voxel_strategy}, patient strategy {patient_strategy}!")
                print(e)


def main(list_cercare_maps,
         list_labels,
         list_features,
         list_reg_tps,
         list_interpolators,
         voxel_strategy,
         patient_strategy):

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_data_results = (constants.DIR_TOTAL_CUTOFFS
                         / voxel_strategy
                         / patient_strategy
                         / fr"Cercare_results_{current_time}.csv")
    path_data_results.parent.mkdir(parents=True, exist_ok=True)

    df_data = pd.DataFrame(columns=["label",
                                    "Feature",
                                    "reg_tp",
                                    "interpolator",
                                    "voxel strategy",
                                    "patient_category",
                                    "Cutoff", "Recall", "Specificity", "total AUC"])

    for imaging in list_cercare_maps:
        for label in list_labels:
            for feature in list_features:
                for reg_tp in list_reg_tps:
                    for interpolator in list_interpolators:
                        create_cercare_total_roc(imaging=imaging,
                                                 label=label,
                                                 reg_tp=reg_tp,
                                                 feature=feature,
                                                 interpolator=interpolator,
                                                 voxel_strategy=voxel_strategy,
                                                 patient_strategy=patient_strategy,
                                                 df_data=df_data)

                        print(f"Total ROC generated for imaging {imaging}, label {label}, feature {feature},"
                              f"reg_tp {reg_tp}, interpolator {interpolator}, voxel strategy {voxel_strategy},"
                              f"patient strategy {patient_strategy}!")

    df_data.to_csv(path_data_results, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create Total ROC curve for Cercare features')

    parser.add_argument('--cercare_maps', nargs='+', default=constants.LIST_CERCARE_MAPS,
                        help='list of Cercare images')

    parser.add_argument('--labels', nargs='+',
                        default=["L3R", "L3R_5x5x5", "L3R - (L1 + L3)", "L3R - (L1 + L3)_5x5x5",
                                 "L1", "L1_5x5x5", "L2", "L2_5x5x5", "L3", "L3_5x5x5",
                                 "L4", "L4_5x5x5", "L5", "L5_5x5x5"], help='list of Labels')

    parser.add_argument('--features', nargs='+', default=[None, "mean_5x5x5"],
                        help='list of Features')

    parser.add_argument('--reg_tps', nargs='+', default=["Affine", "SyN"],
                        help='list of registration types')

    parser.add_argument('--interpolators', nargs='+', default=constants.LIST_INTERPOLATORS,
                        help='list of interpolators')

    parser.add_argument('--voxel_strategy', default="all_voxels",
                        help='Voxel strategy, can be either all_voxels, CTV or OUTSIDE_CTV')

    parser.add_argument('--patient_strategy', default="all",
                        help='Patient strategy, can be either all or Class or surgery_type or IDH')

    args = parser.parse_args()

    features = [None if feature == 'None' else feature for feature in args.features]

    main(list_cercare_maps=args.cercare_maps,
         list_labels=args.labels,
         list_features=features,
         list_reg_tps=args.reg_tps,
         list_interpolators=args.interpolators,
         voxel_strategy=args.voxel_strategy,
         patient_strategy=args.patient_strategy)
