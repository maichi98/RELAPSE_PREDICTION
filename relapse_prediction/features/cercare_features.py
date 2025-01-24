from relapse_prediction.features.features import create_features
from relapse_prediction import constants

import pandas as pd
import numpy as np

 
def get_cercare_features(patient, imaging, interpolator, feature=None, p=None, **kwargs):

    p = constants.DICT_CERCARE_P[imaging] if p is None else p
    feature = f"{imaging}_{feature}" if feature is not None else imaging

    path_features = constants.DIR_FEATURES / patient / fr"{patient}_{imaging}_{interpolator}_features.parquet"
    if not path_features.exists():
        create_features(patient, imaging, **kwargs)

    df_features = pd.read_parquet(path_features, engine="pyarrow")
    if feature not in df_features.columns:
        create_features(patient, imaging, **kwargs)
   
    # Quantize the feature column :
    feature_col = fr"{feature}_quantized" 
    df_features[feature_col] = np.round(df_features[feature], p)
    
    return df_features[["x", "y", "z", feature, feature_col]]
