from relapse_prediction.roc.roc import create_roc
from relapse_prediction import features, labels
from relapse_prediction import constants

from concurrent.futures import ProcessPoolExecutor
import argparse
import os


def create_cercare_roc(patient, imaging, label, feature):

    feature_col = f"{imaging}_{feature}" if feature is not None else imaging
    feature_col = f"{feature_col}_quantized"

    path_roc_results = constants.dir_results / "thresholds per patient" / patient / label / f"{feature_col}.pickle"
    path_roc_plot = constants.dir_results / "ROC per patient" / label / feature_col / f"{patient}.png"

    if path_roc_results.exists() and path_roc_plot.exists():
        print(f"ROC already generated for patient {patient}, label {label}, imaging {imaging} feature {feature} !")
        return

    df_labels = labels.get_df_labels(patient, label)
    df_features = features.get_cercare_features(patient, imaging, feature)
    df_data = df_labels.merge(df_features, on=["x", "y", "z"], how="left")

    create_roc(df_data, patient, label, feature_col)


# ------------------------------------------------ Main functions ------------------------------------------------------
def main(list_mri_maps, list_labels, list_patients, feature):

    for patient in list_patients:
        for imaging in list_mri_maps:
            for label in list_labels:

                create_cercare_roc(patient, imaging, label, feature)
                print(f"ROC generated for patient {patient}, label {label}, imaging {imaging} feature {feature} !")


def process_patient_imaging_label(tpl):
    patient, imaging, label, feature = tpl
    create_cercare_roc(patient, imaging, label, feature)
    print(f"ROC generated for patient {patient}, label {label}, imaging {imaging} feature {feature} !")


def main_mp(list_mri_maps, list_labels, list_patients, feature, num_workers):

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pairs = [(patient, imaging, label, feature)
                 for patient in list_patients
                 for imaging in list_mri_maps
                 for label in list_labels]
        executor.map(process_patient_imaging_label, pairs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create ROC curve for Cercare features')

    parser.add_argument('--cercare_maps', nargs='+', default=constants.L_CERCARE_MAPS,
                        help='list of Cercare images')

    parser.add_argument('--labels', nargs='+',
                        default=["L3R", "L3R_5x5x5", "L3R - (L1 + L3)", "L3R - (L1 + L3)_5x5x5",
                                 "L1", "L1_5x5x5", "L2", "L2_5x5x5", "L3", "L3_5x5x5", "L4", "L4_5x5x5",
                                 "L5", "L5_5x5x5"], help='list of Labels')

    parser.add_argument('--feature', default=None,
                        help="choice of feature")

    parser.add_argument('--start', type=int, default=0,
                        help='start index of the list of patients')

    parser.add_argument('--end', type=int, default=len(constants.list_patients),
                        help='end index of the list of patients')

    parser.add_argument('--mp', action='store_true', default=False,
                        help='Use multiprocessing ?')

    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help='number of CPU workers')

    args = parser.parse_args()

    if not args.mp:
        main(list_mri_maps=args.cercare_maps,
             list_labels=args.labels,
             list_patients=constants.list_patients[args.start: args.end],
             feature=args.feature)
    else:
        main_mp(list_mri_maps=args.cercare_maps,
                list_labels=args.labels,
                list_patients=constants.list_patients[args.start: args.end],
                feature=args.feature,
                num_workers=args.num_workers)
