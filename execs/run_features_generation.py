from relapse_prediction import constants, features, utils

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def process_patient(patient):

    df_mask = utils.get_df_mask(patient)
    for imaging in constants.L_IRM_MAPS:
        df_imaging_features = features.get_df_imaging_features(patient, imaging, df_mask=df_mask)
        print(f"Successfully generated features for patient : {patient} and imaging : {imaging}")


def main():

    with ProcessPoolExecutor(max_workers=20) as executor:
        executor.map(process_patient, constants.list_patients)


if __name__ == "__main__":
    main()
