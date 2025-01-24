from relapse_prediction import utils, constants

from scipy.ndimage import binary_dilation
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import pandas as pd
import argparse
import ants
import os


def add_ctv_label(patient, df_labels):

    list_labels = ["L1", "L2", "L3", "L4", "L5"]

    np_ctv = None
    for label in list_labels:
        path_label = constants.dir_processed / patient / "pre_RT" / "Labels" / f"{patient}_pre_RT_{label}.nii.gz"
        ants_label = ants.image_read(str(path_label))
        np_label = ants_label.numpy()
        np_ctv = np_label if np_ctv is None else np_ctv + np_label

    np_ctv[np_ctv > 0] = 1
    np_ctv[np_ctv < 0] = 0

    # Dilation of the CTV :
    np_ctv = binary_dilation(np_ctv, iterations=3)

    df_ctv = utils.flatten_to_df(np_ctv, "CTV")
    df_labels = df_labels.merge(df_ctv, on=["x", "y", "z"], how="left")
    return df_labels


def add_l3_plus_l3r_label(patient, df_labels):

    df_labels["L3 + L3R"] = df_labels["L3R"] + df_labels["L3"]
    df_labels.loc[df_labels["L3 + L3R"] > 1, "L3 + L3R"] = 1

    df_index = df_labels[["x", "y", "z", "index_5x5x5", "L3 + L3R"]].copy()
    _df_index = df_index.groupby("index_5x5x5").mean("L3 + L3R").reset_index()[["index_5x5x5", "L3 + L3R"]].rename(
        columns={"L3 + L3R": "L3 + L3R_5x5x5"})

    df_index = df_index.merge(_df_index, on="index_5x5x5", how="left").drop(columns="L3 + L3R")
    df_labels = df_labels.merge(df_index, on=["x", "y", "z", "index_5x5x5"], how="left")

    df_labels.loc[df_labels["L3 + L3R_5x5x5"] < 0.5, "L3 + L3R_5x5x5"] = 0
    df_labels.loc[df_labels["L3 + L3R_5x5x5"] >= 0.5, "L3 + L3R_5x5x5"] = 1

    return df_labels


def add_sum_labels(patient, df_labels):

    df_labels["SumPreRT + L3R"] = df_labels["L3"] + df_labels["L1"] + df_labels["L5"] + df_labels["L2"] + df_labels["L3R"]
    df_labels.loc[df_labels["SumPreRT + L3R"] > 1, "SumPreRT + L3R"] = 1

    df_labels["L2 + L3R"] = df_labels["L2"] + df_labels["L3R"]
    df_labels.loc[df_labels["L2 + L3R"] > 1, "L2 + L3R"] = 1
    df_labels.loc[df_labels["L2 + L3R"] < 0, "L2 + L3R"] = 0

    df_labels["L2 + L3R - (L1 + L3)"] = df_labels["L2"] - df_labels["L3R - (L1 + L3)"]
    df_labels.loc[df_labels["L2 + L3R - (L1 + L3)"] > 1, "L2 + L3R - (L1 + L3)"] = 1
    df_labels.loc[df_labels["L2 + L3R - (L1 + L3)"] < 0, "L2 + L3R - (L1 + L3)"] = 0

    return df_labels


def add_new_labels(patient):

    path_labels = constants.dir_labels / f"{patient}_labels.parquet"
    if not path_labels.exists():
        raise FileNotFoundError(f"Labels not found for patient {patient} !")
    df_labels = pd.read_parquet(path_labels, engine="pyarrow")

    # df_labels = add_ctv_label(patient, df_labels)
    # df_labels = add_l3_plus_l3r_label(patient, df_labels)
    df_labels = add_sum_labels(patient, df_labels)

    df_labels.to_parquet(path_labels, engine="pyarrow")


# ------------------------------------------------ Main functions ------------------------------------------------------
def main(list_patients):

    for patient in tqdm(list_patients):
        add_new_labels(patient)
        print(f"New labels added for patient: {patient} !")


def process_patient(patient):
    add_new_labels(patient)
    print(f"New labels added for patient: {patient} !")


def main_mp(list_patients, num_workers):

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_patient, list_patients)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add new labels to the labels dataframe')

    parser.add_argument('--start', type=int, default=0,
                        help='start index of the patients list')

    parser.add_argument('--end', type=int, default=len(constants.list_patients),
                        help='end index of the list of patients')

    parser.add_argument('--mp', action='store_true', default=False,
                        help='Use multiprocessing ?')

    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help='number of CPU workers')

    args = parser.parse_args()

    if not args.mp:
        main(constants.list_patients[args.start: args.end])
    else:
        main_mp(constants.list_patients[args.start: args.end], args.num_workers)



