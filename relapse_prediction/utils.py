from relapse_prediction.constants import *

from torch.nn.functional import conv2d, conv3d
import pandas as pd
import numpy as np
import torch
import ants


def flatten_to_df(arr, col):
    xx, yy, zz = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]), np.arange(arr.shape[2]), indexing='ij')
    flattened_arr = np.vstack((xx.ravel(), yy.ravel(), zz.ravel(), arr.ravel())).T

    return pd.DataFrame(flattened_arr, columns=["x", "y", "z", col])


# A patient's mask corresponds to the pre RT T1 imaging :
def get_df_mask(patient):

    path_prert_t1 = dir_processed / patient / "pre_RT" / "T1" / fr"{patient}_pre_RT_T1.nii.gz"
    ants_prert_t1 = ants.image_read(str(path_prert_t1))
    ants_mask = ants.get_mask(ants_prert_t1)

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
