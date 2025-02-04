from relapse_prediction.roc.roc import create_roc
from relapse_prediction import features, labels
from relapse_prediction import constants, utils

from concurrent.futures import ProcessPoolExecutor
import argparse
import os


def create_mri_roc(patient, imaging, label, reg_tp, feature, norm, voxel_strategy, overwrite=False):

    voxel_strategy = voxel_strategy.upper()

    if voxel_strategy not in ["ALL_VOXELS", "CERCARE_ONLY", "CERCARE_NO_VENTRICLES"]:
        raise ValueError("voxel_strategy must be either 'ALL_VOXELS', 'CERCARE_ONLY' or 'CERCARE_NO_VENTRICLES' !")

    feature_col = f"{imaging}_{feature}" if feature is not None else imaging
    feature_col = f"{feature_col}_{norm}_normalized"

    path_thresholds = (constants.DIR_THRESHOLDS
                       / voxel_strategy
                       / reg_tp
                       / "MRI"
                       / patient
                       / f"{label}_{feature_col}.pickle")

    path_roc_plot = (constants.DIR_ROC_PLOTS
                     / voxel_strategy
                     / reg_tp
                     / "MRI"
                     / label
                     / feature_col
                     / f"{patient}.png")
    title = fr" {patient} : Label : {label} ({reg_tp}) {feature_col}. Voxels : {voxel_strategy}"

    if path_thresholds.exists() and path_roc_plot.exists() and not overwrite:
        # print(f"ROC already generated for patient {patient}, label {label}, imaging {imaging} feature {feature} voxel strategy {voxel_strategy}!")
        return

    df_labels = labels.get_df_labels(patient=patient, label=label, reg_tp=reg_tp)

    if voxel_strategy == "CERCARE_ONLY":
        df_labels = df_labels[df_labels["CERCARE"] == 1]
    elif voxel_strategy == "CERCARE_NO_VENTRICLES":
        df_labels = df_labels[(df_labels["CERCARE"] == 1) & (df_labels["Ventricles"] == 0)]

    df_features = features.get_mri_features(patient, imaging, feature, norm)
    df_data = df_labels.merge(df_features, on=["x", "y", "z"], how="left")

    create_roc(df_data=df_data, label=label, feature_col=feature_col, path_thresholds=path_thresholds,
               path_roc_plot=path_roc_plot, title=title)


# ------------------------------------------------ Main functions ------------------------------------------------------
def main(list_patients, list_mri_maps, list_labels, reg_tp, feature, norm, voxel_strategy, overwrite):

    for patient in list_patients:
        for imaging in list_mri_maps:
            for label in list_labels:
                try:
                    create_mri_roc(patient=patient, imaging=imaging, label=label, reg_tp=reg_tp,
                                   feature=feature, norm=norm, voxel_strategy=voxel_strategy, overwrite=overwrite)
                    print(f"ROC generated for patient :  {patient}, imaging :  {imaging}, label :  {label},"
                          f"reg_tp :  {reg_tp}, feature :  {feature}, norm :  {norm},"
                          f" voxel strategy :  {voxel_strategy}!")

                except Exception as e:
                    print(f"Error for patient :  {patient}, imaging :  {imaging}, label :  {label},"
                          f"reg_tp :  {reg_tp}, feature :  {feature}, norm :  {norm},"
                          f" voxel strategy :  {voxel_strategy}!")
                    print(e)
                    continue


def process_patient_imaging_label(tpl):
    patient, imaging, label, reg_tp, feature, norm, voxel_strategy, overwrite = tpl
    try:
        create_mri_roc(patient=patient, imaging=imaging, label=label, reg_tp=reg_tp,
                       feature=feature, norm=norm, voxel_strategy=voxel_strategy, overwrite=overwrite)
        print(f"ROC generated for patient :  {patient}, imaging :  {imaging}, label :  {label},"
              f"reg_tp :  {reg_tp}, feature :  {feature}, norm :  {norm},"
              f" voxel strategy :  {voxel_strategy}!")
    except Exception as e:
        print(f"Error for patient :  {patient}, imaging :  {imaging}, label :  {label},"
              f"reg_tp :  {reg_tp}, feature :  {feature}, norm :  {norm},"
              f" voxel strategy :  {voxel_strategy}!")
        print(e)


def main_mp(list_patients, list_mri_maps, list_labels, reg_tp, feature, norm, voxel_strategy, overwrite, num_workers):

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pairs = [(patient, imaging, label, reg_tp, feature, norm, voxel_strategy, overwrite)
                 for patient in list_patients
                 for imaging in list_mri_maps
                 for label in list_labels]
        executor.map(process_patient_imaging_label, pairs)


if __name__ == "__main__":

    list_patients = utils.get_perfusion_patients()

    parser = argparse.ArgumentParser(description='Create ROC curve for MRI features')

    parser.add_argument('--mri_maps', nargs='+', default=constants.LIST_MRI_MAPS,
                        help='list of MRI images')

    parser.add_argument('--labels', nargs='+',
                        default=["L3R", "L3R_5x5x5", "L3R - (L1 + L3)", "L3R - (L1 + L3)_5x5x5",
                                 "L1", "L1_5x5x5", "L2", "L2_5x5x5", "L3", "L3_5x5x5", "L4", "L4_5x5x5",
                                 "L5", "L5_5x5x5"], help='list of Labels')

    parser.add_argument('--reg_tp', default="Affine",
                        help='choice of registration type')

    parser.add_argument('--feature',
                        default=None,
                        type=lambda x: None if x.lower() == 'none' else x,
                        help="choice of feature (pass 'None' to get the default None value)")

    parser.add_argument('--norm', type=str, default='z_score',
                        help='normalization method of the features')

    parser.add_argument('--voxel_strategy', default="all_voxels",
                        help="choice of voxel strategy")

    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing ROC curves ?')

    parser.add_argument('--start', type=int, default=0,
                        help='start index of the list of patients')

    parser.add_argument('--end', type=int, default=len(list_patients),
                        help='end index of the list of patients')

    parser.add_argument('--mp', action='store_true', default=False,
                        help='Use multiprocessing ?')

    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help='number of CPU workers')

    args = parser.parse_args()

    if args.reg_tp == "SyN":
        list_patients = utils.get_has_syn_patients()

    list_patients = list_patients[args.start: args.end]

    if not args.mp:
        main(list_patients=list_patients,
             list_mri_maps=args.mri_maps,
             list_labels=args.labels,
             reg_tp=args.reg_tp,
             feature=args.feature,
             norm=args.norm,
             voxel_strategy=args.voxel_strategy,
             overwrite=args.overwrite)
    else:
        main_mp(list_patients=list_patients,
                list_mri_maps=args.mri_maps,
                list_labels=args.labels,
                reg_tp=args.reg_tp,
                feature=args.feature,
                norm=args.norm,
                voxel_strategy=args.voxel_strategy,
                overwrite=args.overwrite,
                num_workers=args.num_workers)
