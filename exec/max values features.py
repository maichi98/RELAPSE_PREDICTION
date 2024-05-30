from relapse_prediction import constants
from pathlib import Path
from tqdm import tqdm
import shutil
import pandas as pd
import os
import pickle
from sklearn.metrics import roc_curve, auc
import numpy as np


if __name__ == "__main__":

    df_volumes = pd.DataFrame(columns=["patient", "(L1 + L3) volume", "L2 volume"])

    for patient in tqdm(constants.list_patients):
        df_labels = pd.read_parquet(constants.dir_labels / f"{patient}_labels.parquet", engine="pyarrow")
        l1_l3_volume = df_labels["(L1 + L3)"].sum() / 1000
        l2_volume = df_labels["L2"].sum() / 1000

        df_volumes.loc[len(df_volumes)] = [patient, l1_l3_volume, l2_volume]

    df_volumes.to_csv(constants.dir_results / "volumes.csv", index=False)




