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


def get_df_labels(patient, label):
    
    path_labels = constants.dir_labels / f"{patient}_labels.parquet"
    if not path_labels.exists():
        create_labels(patient)

    return pd.read_parquet(path_labels, engine="pyarrow")[['x', 'y', 'z', label]]


def create_labels(patient):
    path_labels = constants.dir_labels / f"{patient}_labels.parquet"

    df_mask = utils.get_df_mask(patient)

    dict_labels = [
        ("pre_RT", "L1", "L1"),
        ("pre_RT", "L3", "L3"),
        ("Rechute", "L3", "L3R")
    ]
    df_labels = df_mask.copy()
    for stage, lbl, label in dict_labels:
        path_label = constants.dir_processed / patient / stage / "Labels" / f"{patient}_{stage}_{lbl}_Label.nii.gz"
        ants_label = ants.image_read(str(path_label))

        df_label = utils.flatten_to_df(ants_label.numpy(), label)
        df_labels = df_labels.merge(df_label, on=["x", "y", "z"], how="left")

    # Create L3R - (L1 + L3) label :
    df_labels["L3R - (L1 + L3)"] = df_labels["L3R"] - df_labels["L1"] - df_labels["L3"]
    df_labels.loc[df_labels["L3R - (L1 + L3)"] < 0, "L3R - (L1 + L3)"] = 0

    # add index_5x5x5 : 
    new_shape = (48, 48, 31)
    df_index = index_per_region(df_mask, (240, 240, 155), new_shape)
    df_index.rename(columns={"index_48x48x31": "index_5x5x5"}, inplace=True)
    df_labels = df_labels.merge(df_index, on=['x', 'y', 'z'], how="left")

    for label in ["L3R", "L3R - (L1 + L3)"]:
        df_index = df_labels[["x", "y", "z", "index_5x5x5", label]].copy()
        _df_index = df_index.groupby("index_5x5x5").mean(label).reset_index()[["index_5x5x5", label]].rename(columns={label: f"{label}_5x5x5"})
        df_index = df_index.merge(_df_index, on="index_5x5x5", how="left").drop(columns=(label))
        df_labels = df_labels.merge(df_index, on=["x", "y", "z", "index_5x5x5"], how="left")
        df_labels.loc[df_labels[f"{label}_5x5x5"] < 0.5, f"{label}_5x5x5"] = 0
        df_labels.loc[df_labels[f"{label}_5x5x5"] >= 0.5, f"{label}_5x5x5"] = 1

    path_labels.parent.mkdir(exist_ok=True, parents=True)
    df_labels.to_parquet(path_labels, engine="pyarrow")
