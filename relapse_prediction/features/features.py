from relapse_prediction import constants, utils

from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import argparse
import ants
import os


def create_features(patient: str, imaging: str, **kwargs):
    """
    Create features for a given patient and imaging.

    This function generates features by convolving the imaging data with a set of kernels.
    The features are then flattened to a DataFrame and merged with the mask DataFrame.
    If features already exist, they are read from a parquet file.
    If any kernels are missing from the existing features, the imaging is convolved with these kernels,
    and the resulting features are added to the DataFrame, which is then saved.

    Parameters:
    -----------
    patient : (str)
        The patient's name or ID.

    imaging : (str)
        The imaging sequence's name or ID.

    **kwargs : (dict)
        Additional arguments.

    """

    dir_patient = constants.dir_features / patient
    dir_patient.mkdir(exist_ok=True, parents=True)
    path_features = dir_patient / fr"{patient}_{imaging}_features.parquet"

    if not path_features.exists():
        path_imaging = constants.dir_processed / patient / "pre_RT" / imaging / fr"{patient}_pre_RT_{imaging}.nii.gz"
        ants_imaging = ants.image_read(str(path_imaging))
        _df_features = utils.flatten_to_df(ants_imaging.numpy(), imaging)
        df_features = utils.get_df_mask(patient)
        df_features = df_features.merge(_df_features, on=["x", "y", "z"], how="left")
    else:
        df_features = pd.read_parquet(path_features, engine="pyarrow")

    if "dict_kernels" in kwargs.keys():
        dict_kernels = kwargs["dict_kernels"]
    else:
        dict_kernels = constants.D_KERNELS

    list_kernels = list(dict_kernels.keys())
    list_kernel_cols = [col.lstrip(f"{imaging}_") for col in df_features.columns if col.startswith(f"{imaging}_")]
    list_missing_kernels = list(set(list_kernels) - set(list_kernel_cols))

    for id_kernel in list_missing_kernels:
        kernel = dict_kernels[id_kernel]
        ants_conv_feature = utils.get_convolved_imaging(patient, imaging, id_kernel, kernel, save=True)
        _df_features = utils.flatten_to_df(ants_conv_feature.numpy(), f"{imaging}_{id_kernel}")
        df_features = df_features.merge(_df_features, on=["x", "y", "z"], how='left')

    if len(list_missing_kernels) != 0:
        df_features.to_parquet(str(path_features), engine="pyarrow")


# ---------------------------------------------- Main functions --------------------------------------------------------
def main(list_images: list, list_patients: list):

    for patient in list_patients:
        for imaging in list_images:
            create_features(patient, imaging)
            print(f"Features generated for patient: {patient} for the imaging {imaging} !")


def process_patient_imaging(pair):
    patient, imaging = pair
    create_features(patient, imaging)
    print(f"Features generated for patient: {patient} for the imaging {imaging} !")


def main_mp(list_images: list, list_patients: list, num_workers):

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pairs = [(patient, imaging) for patient in list_patients for imaging in list_images]
        executor.map(process_patient_imaging, pairs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate features")

    parser.add_argument('--maps', nargs='+', default=constants.L_IRM_MAPS + constants.L_CERCARE_MAPS,
                        help='list of feature sequences')
    parser.add_argument('--start', type=int, default=0,
                        help='start index of the list of patients')
    parser.add_argument('--end', type=int, default=len(constants.list_patients),
                        help='end index of the list of patients')
    parser.add_argument('--mp', action='store_true', default=False,
                        help='use multiprocessing')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help='number of workers')

    args = parser.parse_args()

    if not args.mp:
        main(list_images=args.maps,
             list_patients=constants.list_patients[args.start: args.end])
    else:
        main_mp(list_images=args.maps,
                list_patients=constants.list_patients[args.start: args.end],
                num_workers=args.num_workers)
