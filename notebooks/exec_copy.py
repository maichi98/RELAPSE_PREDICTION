from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import argparse
import shutil

import concurrent.futures


dir_processed = Path("/media/maichi/SSD-IGR/AIDREAM_DATA/processed")

df_links = pd.read_csv("/home/maichi/work/my_projects/AIDREAM/RELAPSE_PREDICTION/notebooks/extended_patients.csv")
list_patients = df_links["id_aidream"].loc[df_links["Perfusion directory"] != "Not found"]

parser = argparse.ArgumentParser(description='Copy perfusion data')
parser.add_argument('-i', type=int, default=0, help='fold')

args = parser.parse_args()


def copy_file(patient):
    dir_perfusion = Path(df_links.loc[df_links["id_aidream"] == patient, "Perfusion directory"].values[0])
    dir_dst = Path("/media/maichi/SSD-IGR/Perfusion data") / patient
    dir_dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src=dir_perfusion, dst=dir_dst, dirs_exist_ok=True)


start = args.i * 20
end = min((args.i + 1) * 20, len(list_patients))

list_patients = list_patients[start:end]


if __name__ == "__main__":

    # Use a ThreadPoolExecutor to copy files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        list(executor.map(copy_file, list_patients))
