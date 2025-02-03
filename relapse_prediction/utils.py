from relapse_prediction import constants

from torch.nn.functional import conv2d, conv3d
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import ants


def get_perfusion_patients():

    list_patients = constants.PATH_PERFUSION_PATIENTS.read_text().splitlines()
    list_patients = [list_patients[i]
                     for i in np.argsort([int(p.strip("AIDREAM_"))
                                          for p in list_patients
                                          if p.startswith("AIDREAM_")
                                          ])
                     ]

    return list_patients


def get_imaging(patient: str, imaging: str, is_cercare: bool, interpolator: str = None):

    if is_cercare:
        path_imaging = (constants.DIR_PROCESSED
                        / patient
                        / "CERCARE"
                        / interpolator
                        / fr"{patient}_{imaging}_{interpolator}.nii.gz")
    else:
        path_imaging = (constants.DIR_PROCESSED
                        / patient
                        / "MRI"
                        / fr"{patient}_pre_RT_{imaging}.nii.gz")

    return ants.image_read(str(path_imaging))


def flatten_to_df(arr, col):
    xx, yy, zz = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]), np.arange(arr.shape[2]), indexing='ij')
    flattened_arr = np.vstack((xx.ravel(), yy.ravel(), zz.ravel(), arr.ravel())).T

    return pd.DataFrame(flattened_arr, columns=["x", "y", "z", col])


def get_df_mask(patient):

    path_mask = (constants.DIR_PROCESSED
                 / patient
                 / "MRI"
                 / fr"{patient}_pre_RT_T1_mask.nii.gz")
    assert path_mask.exists(), f"Mask for patient {patient} does not exist !"

    ants_mask = ants.image_read(str(path_mask)) > 0

    df_mask = flatten_to_df(ants_mask.numpy(), "mask")
    df_mask = df_mask.loc[df_mask["mask"] == 1].drop(columns=["mask"]).reset_index(drop=True)
    return df_mask


def convolve(arr, kernel):
    weight = torch.tensor(kernel, dtype=torch.float32)
    input = torch.tensor(arr, dtype=torch.float32)

    if len(kernel.shape) == 2:
        input, weight = input.unsqueeze(1), weight.unsqueeze(0).unsqueeze(0)
        output = conv2d(input, weight, stride=1, padding="same")

    elif len(kernel.shape) == 3:
        input, weight = input.unsqueeze(0).unsqueeze(0), weight.unsqueeze(0).unsqueeze(0)
        output = conv3d(input, weight, stride=1, padding="same")

    else:
        raise ValueError("the Kernel should be 2D, or 3D !")

    return output.squeeze().numpy()


# def get_list_imaging_features(imaging, feature):
#
#     list_features = []
#     for patient in constants.list_patients:
#
#         path_features = constants.dir_features / patient / fr"{patient}_{imaging}_features.parquet"
#         df_features = pd.read_parquet(path_features, engine="pyarrow")
#         list_features += df_features[feature].tolist()
#
#     return list_features

#
def get_convolved_imaging(patient, imaging, id_kernel, kernel, is_cercare: bool, interpolator: str = None, save=True):

    if is_cercare:
        path_imaging_conv = (constants.DIR_HARD_DRIVE
                             / "AIDREAM DATA"
                             / "CERCARE DATA"
                             / "CONVOLVED REGISTERED CERCARE"
                             / patient
                             / interpolator
                             / fr"{patient}_{imaging}_{interpolator}_{id_kernel}.nii.gz")
    else:
        path_imaging_conv = (constants.DIR_HARD_DRIVE
                             / "AIDREAM DATA"
                             / "MRI DATA"
                             / "CONVOLVED REGISTERED MRI"
                             / patient
                             / fr"{patient}_pre_RT_{imaging}_{id_kernel}.nii.gz")

    path_imaging_conv.parent.mkdir(parents=True, exist_ok=True)

    if path_imaging_conv.exists():
        return ants.image_read(str(path_imaging_conv))
    else:
        ants_imaging = get_imaging(patient, imaging, is_cercare, interpolator)
        np_conv_imaging = convolve(ants_imaging.numpy(), kernel)
        ants_conv_imaging = ants_imaging.new_image_like(np_conv_imaging)
        if save:
            ants_conv_imaging.to_file(path_imaging_conv)

        return ants_conv_imaging


def normalize(s, norm):

    if norm == "max":
        max_value = s.max()
        return s / max_value

    if norm == "min_max":
        min_value, max_value = s.min(), s.max()
        return (s - min_value) / (max_value - min_value)

    if norm == "z_score":
        return stats.zscore(s)

    raise ValueError(f"{norm} must be either, min_max, max, or z_score !")


def get_referential_table(list_patients: str = None):

    path_referential_table = constants.PATH_REFERENTIAL_TABLE
    df_referential_table = pd.read_csv(path_referential_table)

    if list_patients is None:
        return df_referential_table

    return df_referential_table[df_referential_table["patient"].isin(list_patients)].reset_index(drop=True)


def get_has_syn_patients():

    df_ref = get_referential_table()
    list_patients = df_ref.loc[df_ref["has SyN"] == 1]["AIDREAM_ID"].tolist()

    list_patients = [list_patients[i]
                     for i in np.argsort([int(p.strip("AIDREAM_"))
                                          for p in list_patients
                                          if p.startswith("AIDREAM_")
                                          ])
                     ]

    return list_patients


def get_list_patients_by_strategy(patient_strategy, reg_tp):

    df_ref = get_referential_table()

    if reg_tp == "SyN":
        df_ref = df_ref.loc[df_ref["has SyN"]]

    if patient_strategy == "all":
        return {"all": df_ref["AIDREAM_ID"].tolist()}

    if patient_strategy == "SyN_patients":
        return {"SyN_patients": get_has_syn_patients()}

    if patient_strategy == "Class":
        return {"Class_1": df_ref.loc[df_ref["Class"] == 1]["AIDREAM_ID"].tolist(),
                "Class_2": df_ref.loc[df_ref["Class"] == 2]["AIDREAM_ID"].tolist(),
                "Class_3": df_ref.loc[df_ref["Class"] == 3]["AIDREAM_ID"].tolist()}

    if patient_strategy == "surgery_type":
        return {"GTR": df_ref.loc[df_ref["surgery_type"] == 0]["AIDREAM_ID"].tolist(),
                "STR": df_ref.loc[df_ref["surgery_type"] == 1]["AIDREAM_ID"].tolist(),
                "Biopsy": df_ref.loc[df_ref["surgery_type"] == 2]["AIDREAM_ID"].tolist()}

    raise ValueError(f"patient strategy {patient_strategy} is not valid,"
                     f" must be either all, SyN_patients, Class or surgery_type !")
