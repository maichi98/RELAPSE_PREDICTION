from relapse_prediction import utils, labels, constants


def main():
    for patient in constants.list_patients:
        df_mask = utils.get_df_mask(patient)
        df_labels = labels.get_df_labels(patient, df_mask)


if __name__ == "__main__":
    main()
