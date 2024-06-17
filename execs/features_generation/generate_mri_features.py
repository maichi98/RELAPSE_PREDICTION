from relapse_prediction import constants, features, utils

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse


def process_patient(patient):

    df_mask = utils.get_df_mask(patient)
    for imaging in constants.L_IRM_MAPS:
        df_imaging_features = features.get_df_imaging_features(patient, imaging, df_mask=df_mask)
        print(f"Successfully generated features for patient : {patient} and imaging : {imaging}")


def main(list_mri_maps, list_patients):

    for patient in tqdm(list_patients):
        for imaging in list_mri_maps:
            features.create_mri_features(patient, imaging)
            print(f"Features generated for patient: {patient} for the imaging {imaging} !")


def main_mp(list_mri_maps, list_patients, num_workers):
    pass
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MRI features")

    parser.add_argument('--mri_images', nargs='+', default=constants.L_IRM_MAPS, help='list of MRI images')
    parser.add_argument('--start', type=int, default=0, help='start index of the list of patients')
    parser.add_argument('--end', type=int, default=len(constants.list_patients), help='end index of the list of patients')
    parser.add_argument('--mp', action='store_true', default=False, help='Use multiprocessing ?')
    parser.add_argument('--num_workers', type=int, default=10, help='number of CPU workers')

    args = parser.parse_args()

    list_mri_maps = args.mri_images
    list_patients = constants.list_patients[args.start: args.end]

    if not args.mp:
        main(list_mri_maps, list_patients)
    else:
        main_mp(list_mri_maps, list_patients, num_workers)
