from relapse_prediction.features.features import create_features
from relapse_prediction import constants, utils

import pandas as pd
import numpy as np


def get_mri_features(patient, imaging, feature=None, norm=None, p=2, **kwargs):

    # Default normalization is z_score :
    if norm is None:
        norm = "z_score"
    feature = f"{imaging}_{feature}" if feature is not None else imaging

    path_features = constants.dir_features / patient / fr"{patient}_{imaging}_features.parquet"
    if not path_features.exists():
        create_features(patient, imaging, **kwargs)

    df_features = pd.read_parquet(path_features, engine="pyarrow")
    if feature not in df_features.columns:
        create_features(patient, imaging, **kwargs)

    # Normalize the feature column : 
    feature_col = fr"{feature}_{norm}_normalized"
    df_features[feature_col] = utils.normalize(df_features[feature], norm)
    # Quantize the feature column :
    df_features[feature_col] = np.round(df_features[feature_col], p)
    return df_features[["x", "y", "z", feature, feature_col]]
