from relapse_prediction import constants, labels

from tqdm import tqdm
import argparse


def main(list_patients):

    for patient in tqdm(list_patients):
        labels.create_labels(patient)
        print(f"Labels generated for patient: {patient} !")


def main_mp(list_patients, num_workers):
    pass
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Labels")

    parser.add_argument('--start', type=int, default=0, help='start index of the list of patients')
    parser.add_argument('--end', type=int, default=len(constants.list_patients), help='end index of the list of patients')
    parser.add_argument('--mp', action='store_true', default=False, help='Use multiprocessing ?')
    parser.add_argument('--num_workers', type=int, default=10, help='number of CPU workers')

    args = parser.parse_args()

    list_patients = constants.list_patients[args.start: args.end]

    if not args.mp:
        main(list_patients)
    else:
        main_mp(list_patients, num_workers)
