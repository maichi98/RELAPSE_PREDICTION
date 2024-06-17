from relapse_prediction import constants

import argparse


def get_list_thresholds(imaging, label, feature):



def main(list_mri_maps, list_labels, feature, norm):

    for imaging in list_mri_maps:
        for label in list_labels:
            print("get list of thresholds ...")
            list_thresholds = get_list_thresholds(imaging, label, feature)






def main_mp():
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate total ROC curves")

    parser.add_argument('--mri_images', nargs='+', default=constants.L_IRM_MAPS, help='list of MRI images')
    parser.add_argument('--labels', nargs='+', default=["L3R", "L3R_5x5x5", "L3R - (L1 + L3)", "L3R - (L1 + L3)_5x5x5"], help='list of Labels')
    parser.add_argument('--feature', default="None", help="choice of feature")
    parser.add_argument('--norm', type=str, default='z_score', help='normalization method of the features')
    parser.add_argument('--mp', action='store_true', default=False, help='Use multiprocessing ?')
    parser.add_argument('--num_workers', type=int, default=10, help='number of CPU workers')ù

    args = parser.parse_args()

    list_mri_maps = args.mri_images
    list_labels = args.labels
    feature = None if args.feature == "None" else args.feature

    if not args.mp:
        main(list_mri_maps, list_labels, feature, args.norm)
    else:
        main_mp()