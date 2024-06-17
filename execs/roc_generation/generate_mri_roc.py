from relapse_prediction import constants, roc

from tqdm import tqdm
import argparse


def main(list_mri_maps, list_labels, list_patients, feature, norm):

    for patient in tqdm(list_patients):
        for imaging in list_mri_maps:
            for label in list_labels:
                
                roc.create_mri_roc(patient, imaging, label, feature, norm)
                print(fr"Patient: {patient}, label: {label}, imaging: {imaging}, feature: {feature}, norm: {norm} has been processed !")
                

def main_mp(list_mri_maps, list_labels, list_patients, num_workers):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MRI ROC curves")

    parser.add_argument('--mri_images', nargs='+', default=constants.L_IRM_MAPS, help='list of MRI images')
    parser.add_argument('--labels', nargs='+', default=["L3R", "L3R_5x5x5", "L3R - (L1 + L3)", "L3R - (L1 + L3)_5x5x5"], help='list of Labels')
    parser.add_argument('--feature', default="None", help="choice of feature")
    parser.add_argument('--norm', type=str, default='z_score', help='normalization method of the features')
    parser.add_argument('--start', type=int, default=0, help='start index of the list of patients')
    parser.add_argument('--end', type=int, default=len(constants.list_patients), help='end index of the list of patients')
    parser.add_argument('--mp', action='store_true', default=False, help='Use multiprocessing ?')
    parser.add_argument('--num_workers', type=int, default=10, help='number of CPU workers')

    args = parser.parse_args()

    list_patients = constants.list_patients[args.start: args.end]
    list_mri_maps = args.mri_images
    list_labels = args.labels
    feature = None if args.feature == "None" else args.feature

    if not args.mp:
        main(list_mri_maps, list_labels, list_patients, feature, args.norm)
    else:
        main_mp(list_mri_maps, list_labels, list_patients, args.num_workers)
