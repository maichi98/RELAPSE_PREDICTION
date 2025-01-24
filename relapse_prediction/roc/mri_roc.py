from relapse_prediction.roc.roc import create_roc
from relapse_prediction import features, labels
from relapse_prediction import constants

from concurrent.futures import ProcessPoolExecutor
import argparse
import os


def create_mri_roc(patient, imaging, label, feature, norm, voxel_strategy, overwrite=False):

    feature_col = f"{imaging}_{feature}" if feature is not None else imaging
    feature_col = f"{feature_col}_{norm}_normalized"

    path_thresholds = constants.dir_thresholds / voxel_strategy / patient / label / f"{feature_col}.pickle"
    path_roc_plot = constants.dir_roc / voxel_strategy / label / feature_col / f"{patient}.png"

    if path_thresholds.exists() and path_roc_plot.exists() and not overwrite:
        # print(f"ROC already generated for patient {patient}, label {label}, imaging {imaging} feature {feature} voxel strategy {voxel_strategy}!")
        return

    df_labels = labels.get_df_labels(patient, label)

    if voxel_strategy == "CTV":
        df_labels = df_labels[df_labels["CTV"] == 1]
    elif voxel_strategy == "OUTSIDE_CTV":
        df_labels = df_labels[df_labels["CTV"] == 0]

    df_features = features.get_mri_features(patient, imaging, feature, norm)
    df_data = df_labels.merge(df_features, on=["x", "y", "z"], how="left")

    create_roc(df_data, patient, label, feature_col, voxel_strategy)


# ------------------------------------------------ Main functions ------------------------------------------------------
def main(list_mri_maps, list_labels, list_patients, feature, norm, voxel_strategy, overwrite):

    for patient in list_patients:
        for imaging in list_mri_maps:
            for label in list_labels:
                try:
                    create_mri_roc(patient, imaging, label, feature, norm, voxel_strategy, overwrite)
                except Exception as e:
                    print(f"Error for patient {patient}, label {label}, imaging {imaging}, feature {feature}, norm {norm} "
                          f"voxel strategy {voxel_strategy} !")
                    print(e)
                    continue
                print(f"ROC generated for patient {patient}, label {label}, imaging {imaging},"
                      f" feature {feature}, norm {norm} voxel strategy {voxel_strategy}!")


def process_patient_imaging_label(tpl):
    patient, imaging, label, feature, norm, voxel_strategy, overwrite = tpl
    create_mri_roc(patient, imaging, label, feature, norm, voxel_strategy, overwrite)
    print(f"ROC generated for patient {patient}, label {label}, imaging {imaging},"
          f" feature {feature}, norm {norm} voxel strategy {voxel_strategy}!")


def main_mp(list_mri_maps, list_labels, list_patients, feature, norm, voxel_strategy, num_workers, overwrite):

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pairs = [(patient, imaging, label, feature, norm, voxel_strategy, overwrite)
                 for patient in list_patients
                 for imaging in list_mri_maps
                 for label in list_labels]
        executor.map(process_patient_imaging_label, pairs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create ROC curve for MRI features')

    parser.add_argument('--mri_maps', nargs='+', default=constants.L_IRM_MAPS,
                        help='list of MRI images')

    parser.add_argument('--labels', nargs='+',
                        default=["L3R", "L3R_5x5x5", "L3R - (L1 + L3)", "L3R - (L1 + L3)_5x5x5",
                                 "L1", "L1_5x5x5", "L2", "L2_5x5x5", "L3", "L3_5x5x5", "L4", "L4_5x5x5",
                                 "L5", "L5_5x5x5"], help='list of Labels')

    parser.add_argument('--feature', default=None,
                        help="choice of feature")

    parser.add_argument('--norm', type=str, default='z_score',
                        help='normalization method of the features')

    parser.add_argument('--voxel_strategy', default="all_voxels",
                        help="choice of voxel strategy")

    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing ROC curves ?')

    parser.add_argument('--start', type=int, default=0,
                        help='start index of the list of patients')

    parser.add_argument('--end', type=int, default=len(constants.list_patients),
                        help='end index of the list of patients')

    parser.add_argument('--mp', action='store_true', default=False,
                        help='Use multiprocessing ?')

    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help='number of CPU workers')

    args = parser.parse_args()
    list_patients = constants.list_patients[args.start: args.end]

    if not args.mp:
        main(list_mri_maps=args.mri_maps,
             list_labels=args.labels,
             list_patients=list_patients,
             feature=args.feature,
             norm=args.norm,
             voxel_strategy=args.voxel_strategy,
             overwrite=args.overwrite)
    else:
        main_mp(list_mri_maps=args.mri_maps,
                list_labels=args.labels,
                list_patients=list_patients,
                feature=args.feature,
                norm=args.norm,
                num_workers=args.num_workers,
                voxel_strategy=args.voxel_strategy,
                overwrite=args.overwrite)
