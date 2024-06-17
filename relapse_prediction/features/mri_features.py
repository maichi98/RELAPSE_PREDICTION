from relapse_prediction import constants, utils

import pandas as pd
import numpy as np
import ants


def get_mri_features(patient, imaging, feature=None, norm=None, p=2, **kwargs):

    # Default normalization is z_score :
    if norm is None:
        norm = "z_score"
       
    feature = f"{imaging}_{feature}" if feature is not None else imaging

    path_features = constants.dir_features / patient / fr"{patient}_{imaging}_features.parquet"
    if not path_features.exists():
        create_imaging_features(patient, imaging, **kwargs)

    df_features = pd.read_parquet(path_features, engine="pyarrow")
    if feature not in df_features.columns:
        create_mri_features(patient, imaging, **kwargs)

    # Normalize the feature column : 
    feature_col = fr"{feature}_{norm}_normalized"
    df_features[feature_col] = utils.normalize(df_features[feature], norm)
    
    # Quantize the feature column :
    df_features[feature_col] = np.round(df_features[feature_col], p)
    
    return df_features[["x", "y", "z", feature, feature_col]]


def create_mri_features(patient, imaging, **kwargs):

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
