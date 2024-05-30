from relapse_prediction import utils, constants

import pandas as pd
import numpy as np
import ants
import os


def index_per_region(df, shape, new_shape):
    df_reindex = df.copy()

    name_col = f"index_{'x'.join(np.array(new_shape, dtype=str))}"

    delta = [np.ceil(shape[i] / new_shape[i]) for i in range(3)]

    df_reindex[name_col] = 1 + np.floor(df_reindex["x"] / delta[0])
    df_reindex[name_col] += np.floor(df_reindex["y"] / delta[1]) * new_shape[0]
    df_reindex[name_col] += np.floor(df_reindex["z"] / delta[2]) * new_shape[0] * new_shape[1]

    return df_reindex[["x", "y", "z", name_col]]


def get_df_labels(patient, df_mask=None, save=True, **kwargs):
    path_labels = constants.dir_labels / f"{patient}_labels.parquet"

    if path_labels.exists():
        return pd.read_parquet(str(path_labels), engine="pyarrow")

    else:
        # Load selected labels :
        d_labels = [
            ('pre_RT', 'L1', 'L1'),
            ('pre_RT', 'L2', 'L2'),
            ('pre_RT', 'L3', 'L3'),
            ('pre_RT', 'L4', 'L4'),
            ('pre_RT', 'L5', 'L5'),
            ('Rechute', 'L3', 'L3R')
        ]

        df_labels = df_mask.copy()

        for (stage, label, label_name) in d_labels:
            path_label = constants.dir_processed / patient / stage / "Labels" / f"{patient}_{stage}_{label}_Label.nii.gz"
            ants_label = ants.image_read(str(path_label))

            df_label = utils.flatten_to_df(ants_label.numpy(), label_name)
            df_labels = df_labels.merge(df_label, on=['x', 'y', 'z'], how="left")

        # create L3R - (L1 + L3) label :
        df_labels["L3R - (L1 + L3)"] = df_labels["L3R"] - df_labels["L1"] - df_labels["L3"]
        df_labels.loc[df_labels["L3R - (L1 + L3)"] < 0, "L3R - (L1 + L3)"] = 0

        # create (L1 + L3) label :
        df_labels["(L1 + L3)"] = df_labels["L1"] + df_labels["L3"]
        df_labels.loc[df_labels["(L1 + L3)"] > 1, "(L1 + L3)"] = 1

        # create (L4 + L5) label :
        df_labels["(L4 + L5)"] = df_labels["L4"] + df_labels["L5"]
        df_labels.loc[df_labels["(L4 + L5)"] > 1, "(L4 + L5)"] = 1

        # Add index_5x5x5 :
        new_shape = (48, 48, 31)

        df_index = index_per_region(df_mask, (240, 240, 155), new_shape)

        df_index.rename(columns={"index_48x48x31": "index_5x5x5"}, inplace=True)

        df_labels = df_labels.merge(df_index, on=['x', 'y', 'z'], how="left")

        for label in ["L3R", "L3R - (L1 + L3)", "(L1 + L3)", "L2", "L3", "L4", "L5", "(L4 + L5)"]:
            name_index_col, name_mean_col = f"index_5x5x5", f"mean_{label}_5x5x5"

            df_index = df_labels[["x", "y", "z", "index_5x5x5", label]].copy()

            _df_index = df_index.groupby(name_index_col).mean(label).reset_index()[[name_index_col, label]].rename(
                columns={label: name_mean_col})

            df_index = df_index.merge(_df_index, on=name_index_col, how="left").drop(columns=(label))

            df_labels = df_labels.merge(df_index, on=["x", "y", "z", name_index_col], how="left")

        if save:
            if not os.path.exists(constants.dir_labels):
                os.makedirs(constants.dir_labels)
            df_labels.to_parquet(str(path_labels), engine="pyarrow")
        return df_labels
