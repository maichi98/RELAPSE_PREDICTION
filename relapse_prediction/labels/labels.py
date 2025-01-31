from relapse_prediction import utils, constants

from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from scipy.ndimage import binary_dilation
import pandas as pd
import numpy as np
import argparse
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


def get_df_labels(patient, label, reg_tp):
    
    path_labels = (constants.DIR_LABELS
                   / patient
                   / fr"{patient}_labels_{reg_tp}.parquet")

    if not path_labels.exists():
        create_labels(patient, reg_tp)

    return pd.read_parquet(path_labels, engine="pyarrow")[['x', 'y', 'z', 'CERCARE', 'Ventricles', 'CTV', label]]


def create_labels(patient: str, reg_tp: str):

    if reg_tp not in ["SyN", "Affine"]:
        raise ValueError("reg_tp must be either 'SyN' or 'Affine' !")

    path_labels = (constants.DIR_LABELS
                   / patient
                   / fr"{patient}_labels_{reg_tp}.parquet")
    path_labels.parent.mkdir(exist_ok=True, parents=True)

    # First, get the spatial coordinates dataframe :
    df_mask = utils.get_df_mask(patient)

    # Then, add the cercare mask label :
    df_labels = add_cercare_label(patient, df_mask)

    # Add the ventricles mask label :
    df_labels = add_ventricles_label(patient, df_labels)

    # Add the CTV label :
    df_labels = add_ctv_label(patient, df_labels, reg_tp)

    # Add the labels :
    dict_labels = [
        ("pre_RT", "L1"), ("pre_RT", "L2"), ("pre_RT", "L3"), ("pre_RT", "L4"), ("pre_RT", "L5"),
        ("Rechute", "L3R")
    ]

    for stage, label in dict_labels:

        path_label = (constants.DIR_PROCESSED
                      / patient
                      / "LABELS"
                      / stage
                      / reg_tp
                      / fr"{patient}_{stage}_{label}_{reg_tp}.nii.gz")
        assert path_label.exists(), f"Label {label} for patient {patient} does not exist !"

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

    for label in ["L3R", "L3R - (L1 + L3)", "L1", "L2", "L3", "L4", "L5"]:
        df_index = df_labels[["x", "y", "z", "index_5x5x5", label]].copy()
        _df_index = df_index.groupby("index_5x5x5").mean(label).reset_index()[["index_5x5x5", label]].rename(columns={
            label: f"{label}_5x5x5"})
        df_index = df_index.merge(_df_index, on="index_5x5x5", how="left").drop(columns=(label))
        df_labels = df_labels.merge(df_index, on=["x", "y", "z", "index_5x5x5"], how="left")
        df_labels.loc[df_labels[f"{label}_5x5x5"] < 0.5, f"{label}_5x5x5"] = 0
        df_labels.loc[df_labels[f"{label}_5x5x5"] >= 0.5, f"{label}_5x5x5"] = 1

    # Add L3 + L3R label :
    df_labels = add_l3_plus_l3r_label(df_labels)

    # Add sum labels :
    df_labels = add_sum_labels(df_labels)

    path_labels.parent.mkdir(exist_ok=True, parents=True)
    df_labels.to_parquet(path_labels, engine="pyarrow")


def add_cercare_label(patient, df_labels):

    path_cercare_mask = (constants.DIR_PROCESSED
                         / patient
                         / "CERCARE"
                         / fr"{patient}_CERCARE_brainmask_genericLabel.nii.gz")
    assert path_cercare_mask.exists(), f"CERCARE mask for patient {patient} does not exist !"

    ants_cercare_mask = ants.image_read(str(path_cercare_mask))
    df_cercare_mask = utils.flatten_to_df(ants_cercare_mask.numpy(), "CERCARE")
    df_labels = df_labels.merge(df_cercare_mask, on=["x", "y", "z"], how="left")

    return df_labels


def add_ventricles_label(patient, df_labels):

    path_ventricles_mask = (constants.DIR_PROCESSED
                            / patient
                            / "MRI"
                            / fr"{patient}_pre_RT_T1_ventricle_mask.nii.gz")
    assert path_ventricles_mask.exists(), f"Ventricles mask for patient {patient} does not exist !"

    ants_ventricles_mask = ants.image_read(str(path_ventricles_mask))
    df_ventricles_mask = utils.flatten_to_df(ants_ventricles_mask.numpy(), "Ventricles")
    df_labels = df_labels.merge(df_ventricles_mask, on=["x", "y", "z"], how="left")

    return df_labels


def add_ctv_label(patient, df_labels, reg_tp):

    list_labels = ["L1", "L2", "L3", "L4", "L5"]

    np_ctv = None
    for label in list_labels:
        path_label = (constants.DIR_PROCESSED
                      / patient
                      / "LABELS"
                      / "pre_RT"
                      / reg_tp
                      / f"{patient}_pre_RT_{label}_{reg_tp}.nii.gz")
        assert path_label.exists(), f"Label {label} {reg_tp} for patient {patient} does not exist !"

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


def add_l3_plus_l3r_label(df_labels):

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


def add_sum_labels(df_labels):

    df_labels["SumPreRT + L3R"] = df_labels["L3"] + df_labels["L1"] + df_labels["L5"] + df_labels["L2"] + df_labels["L3R"]
    df_labels.loc[df_labels["SumPreRT + L3R"] > 1, "SumPreRT + L3R"] = 1

    df_labels["L2 + L3R"] = df_labels["L2"] + df_labels["L3R"]
    df_labels.loc[df_labels["L2 + L3R"] > 1, "L2 + L3R"] = 1
    df_labels.loc[df_labels["L2 + L3R"] < 0, "L2 + L3R"] = 0

    df_labels["L2 + L3R - (L1 + L3)"] = df_labels["L2"] - df_labels["L3R - (L1 + L3)"]
    df_labels.loc[df_labels["L2 + L3R - (L1 + L3)"] > 1, "L2 + L3R - (L1 + L3)"] = 1
    df_labels.loc[df_labels["L2 + L3R - (L1 + L3)"] < 0, "L2 + L3R - (L1 + L3)"] = 0

    return df_labels


# ------------------------------------------- Main functions -----------------------------------------------------------
def main(list_patients, reg_tp="Affine"):

    for patient in tqdm(list_patients):
        try:
            create_labels(patient, reg_tp)
            print(f"Labels generated for patient: {patient} {reg_tp}!")

        except Exception as e:
            print(f"Error for patient {patient} : {e}")
            continue


def process_patient(pair):

    patient, reg_tp = pair

    try:
        print(fr"Parsing patient {patient} {reg_tp} ...")
        create_labels(patient, reg_tp)
        print(f"Labels generated for patient: {patient} {reg_tp}!")
    except Exception as e:
        print(f"Error for patient {patient} : {e} !")


def main_mp(list_patients, reg_tp, num_workers):

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pairs = [(patient, reg_tp)
                 for patient in list_patients]
        executor.map(process_patient, pairs)


if __name__ == "__main__":

    list_patients = utils.get_perfusion_patients()

    parser = argparse.ArgumentParser(description="Generate Labels")

    parser.add_argument('--start', type=int, default=0,
                        help='start index of the list of patients')
    parser.add_argument('--end', type=int, default=len(list_patients),
                        help='end index of the list of patients')
    parser.add_argument('--reg_tp', type=str, default="Affine",
                        help='registration type (Affine or SyN)')
    parser.add_argument('--mp', action='store_true', default=False,
                        help='Use multiprocessing ?')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help='number of CPU workers')

    args = parser.parse_args()

    list_patients = list_patients[args.start: args.end]

    if args.mp:
        main_mp(list_patients=list_patients,
                reg_tp=args.reg_tp,
                num_workers=args.num_workers)
    else:
        main(list_patients=list_patients,
             reg_tp=args.reg_tp)
