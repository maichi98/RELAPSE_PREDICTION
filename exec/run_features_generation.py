from relapse_prediction import constants, features, utils

import ants


def main():
    for patient in constants.list_patients:
        for imaging in constants.L_CERCARE_MAPS + constants.L_IRM_MAPS:
            dir_imaging = constants.dir_processed / patient / "pre_RT" / imaging

            df_mask = utils.get_df_mask(patient)

            path_imaging = dir_imaging / f"{patient}_pre_RT_{imaging}.nii.gz"

            ants_imaging = ants.image_read(str(path_imaging))

            df_features = features.get_df_imaging_features(patient, imaging, df_mask, constants.D_KERNELS, ants_imaging=ants_imaging)


if __name__ == "__main__":
    main()
