from relapse_prediction import constants
from relapse_prediction import utils

import pandas as pd
import ants
import glob
import os


def get_imaging_conv(patient, imaging, id_kernel, kernel, save=True, **kwargs):
    dir_imaging = constants.dir_processed / patient / "pre_RT" / imaging

    dir_imaging_conv = dir_imaging / "convs"
    dir_imaging_conv.mkdir(exist_ok=True)

    path_imaging_conv = dir_imaging_conv / fr"{patient}_{imaging}_{id_kernel}.nii.gz"

    if path_imaging_conv.exists():
        return ants.image_read(str(path_imaging_conv))
    else:
        if "ants_imaging" in kwargs.keys():
            ants_imaging = kwargs["ants_imaging"]
        elif "path_imaging" in kwargs.keys():
            ants_imaging = ants.image_read(str(kwargs["path_imaging"]))
        else:

            path_imaging = dir_imaging / fr"{patient}_pre_RT_{imaging}.nii.gz"
            ants_imaging = ants.image_read(str(path_imaging))

        np_conv_imaging = utils.convolve(ants_imaging.numpy(), kernel)

        ants_conv_imaging = ants_imaging.new_image_like(np_conv_imaging)
        if save:
            ants_conv_imaging.to_file(path_imaging_conv)
        return ants_conv_imaging


def get_df_imaging_features(patient, imaging, df_mask=None, d_kernels=constants.D_KERNELS, save=True, **kwargs):

    # df_imaging_features columns :
    list_base_cols = ["x", "y", "z", imaging]
    list_kernels = list(d_kernels.keys())
    list_cols = list_base_cols + list_kernels

    # patient features directory:
    dir_patient_features = constants.dir_features / patient
    dir_patient_features.mkdir(exist_ok=True)

    path_imaging_features = dir_patient_features / f"{patient}_{imaging}_features.parquet"

    if path_imaging_features.exists():

        df_features = pd.read_parquet(str(path_imaging_features), engine="pyarrow")
        return df_features[list_cols]

        # n_missing_base_cols = len(set(list_base_cols) - set(df_features.columns))
        # if n_missing_base_cols > 0:
        #     raise ValueError(f"{path_imaging_features} has missing base columns ! ")

    else:
        if "ants_imaging" in kwargs.keys():
            ants_imaging = kwargs["ants_imaging"]
        elif "path_imaging" in kwargs.keys():
            ants_imaging = ants.image_read(str(kwargs["path_imaging"]))
        else:
            path_imaging = constants.dir_processed / patient / "pre_RT" / imaging / f"{patient}_pre_RT_{imaging}.nii.gz"
            ants_imaging = ants.image_read(str(path_imaging))

        _df_features = utils.flatten_to_df(ants_imaging.numpy(), imaging)
        df_features = df_mask.copy()
        df_features = df_features.merge(_df_features, on=["x", "y", "z"], how="left")

    list_missing_kernels = list(set(list_kernels) - set(df_features.columns))

    for id_kernel in list_missing_kernels:
        kernel = d_kernels[id_kernel]
        ants_conv_feature = get_imaging_conv(patient, imaging, id_kernel, kernel, save)
        _df_features = utils.flatten_to_df(ants_conv_feature.numpy(), id_kernel)
        df_features = df_features.merge(_df_features, on=["x", "y", "z"], how='left')

    if save and len(list_missing_kernels) != 0:
        df_features.to_parquet(str(path_imaging_features), engine="pyarrow")
    return df_features[list_cols]


def get_df_features(patient):
    df_features = pd.DataFrame()

    for pq_features in glob.glob(os.path.join(constants.dir_features, patient, f"{patient}_*_features.parquet")):
        imaging = os.path.basename(pq_features).removeprefix(f"{patient}_").removesuffix("_features.parquet")

        _df_features = pd.read_parquet(pq_features, engine="pyarrow")

        L_cols_to_rename = list(set(_df_features.columns) - {"x", "y", "z", imaging})
        _df_features = _df_features.rename(columns={col: f"{imaging}_{col}" for col in L_cols_to_rename})

        df_features = _df_features.copy() if df_features.empty else df_features.merge(_df_features, on=["x", "y", "z"],
                                                                                      how="left")

    return df_features
