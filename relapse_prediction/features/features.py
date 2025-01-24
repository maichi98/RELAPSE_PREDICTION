from relapse_prediction import constants, utils

from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import argparse
import ants
import os


def create_features(patient: str, imaging: str, interpolator: str = None, **kwargs):
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

    if imaging not in constants.LIST_MRI_MAPS + constants.LIST_CERCARE_MAPS:
        raise ValueError(f"Imaging {imaging} not supported !")

    is_cercare = (imaging in constants.LIST_CERCARE_MAPS)
    if is_cercare and interpolator is None:
        raise ValueError(f"Interpolator must be specified for CERCARE maps !")

    suffix = f"{interpolator}_features" if is_cercare else "features"
    path_features = constants.DIR_FEATURES / patient / fr"{patient}_{imaging}_{suffix}.parquet"
    path_features.parent.mkdir(exist_ok=True, parents=True)

    if not path_features.exists():

        ants_imaging = utils.get_imaging(patient=patient, imaging=imaging,
                                         interpolator=interpolator, is_cercare=is_cercare)
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
        ants_conv_feature = utils.get_convolved_imaging(patient, imaging, id_kernel, kernel, save=True,
                                                        interpolator=interpolator, is_cercare=is_cercare)
        _df_features = utils.flatten_to_df(ants_conv_feature.numpy(), f"{imaging}_{id_kernel}")
        df_features = df_features.merge(_df_features, on=["x", "y", "z"], how='left')

    if len(list_missing_kernels) != 0:
        df_features.to_parquet(str(path_features), engine="pyarrow")


# ---------------------------------------------- Main functions --------------------------------------------------------
def main(list_mri_maps: list, list_cercare_maps: list, list_patients: list, list_interpolators: list = None):

    for patient in list_patients:
        for imaging in list_mri_maps:
            print(f"Generating features for patient: {patient} for the imaging {imaging} !")
            create_features(patient, imaging)
            print(f"Features generated for patient: {patient} for the imaging {imaging} !")

        for imaging in list_cercare_maps:
            for interpolator in list_interpolators:
                print(f"Generating features for patient: {patient} for the imaging {imaging} and interpolator {interpolator} !")
                create_features(patient, imaging, interpolator)
                print(f"Features generated for patient: {patient} for the imaging {imaging} and interpolator {interpolator} !")


def process_mri(pair):

    patient, imaging = pair
    print(f"Generating features for patient: {patient} for the imaging {imaging} !")
    create_features(patient, imaging)
    print(f"Features generated for patient: {patient} for the imaging {imaging} !")


def process_cercare(pair):

    patient, imaging, interpolator = pair
    print(f"Generating features for patient: {patient} for the imaging {imaging} and interpolator {interpolator} !")
    create_features(patient, imaging, interpolator)
    print(f"Features generated for patient: {patient} for the imaging {imaging} and interpolator {interpolator} !")


def main_mp(list_mri_maps: list, list_cercare_maps: list, list_patients: list, list_interpolators: list = None,
            num_workers: int = 4):

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for patient in list_patients:
            for imaging in list_mri_maps:
                executor.submit(process_mri, (patient, imaging))

            for imaging in list_cercare_maps:
                for interpolator in list_interpolators:
                    executor.submit(process_cercare, (patient, imaging, interpolator))


if __name__ == "__main__":

    list_patients = utils.get_perfusion_patients()

    parser = argparse.ArgumentParser(description="Generate features")

    parser.add_argument('--mri_maps', nargs='+', default=constants.LIST_MRI_MAPS,
                        help='list of MRI maps')
    parser.add_argument('--cercare_maps', nargs='+', default=constants.LIST_CERCARE_MAPS,
                        help='list of CERCARE maps')
    parser.add_argument('--interpolators', nargs='+', default=constants.LIST_INTERPOLATORS,
                        help='list of interpolators')

    parser.add_argument('--start', type=int, default=0,
                        help='start index of the list of patients')
    parser.add_argument('--end', type=int, default=len(list_patients),
                        help='end index of the list of patients')
    parser.add_argument('--mp', action='store_true', default=False,
                        help='use multiprocessing')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')

    args = parser.parse_args()
    list_patients = list_patients[args.start:args.end]

    if args.mp:
        main_mp(list_mri_maps=args.mri_maps,
                list_cercare_maps=args.cercare_maps,
                list_patients=list_patients,
                list_interpolators=args.interpolators,
                num_workers=args.num_workers)
    else:
        main(list_mri_maps=args.mri_maps,
             list_cercare_maps=args.cercare_maps,
             list_patients=list_patients,
             list_interpolators=args.interpolators)
