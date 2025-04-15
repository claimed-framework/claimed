import argparse
import os
import shutil
from pathlib import Path

import yaml


def identify_corrupted(filestore_root_dir, experiment_id, delete=False):
    experiment_dir = os.path.join(filestore_root_dir, str(experiment_id))
    for run_dir in [elem for elem in os.listdir(experiment_dir) if elem != "meta.yaml"]:
        meta_file_path = os.path.join(experiment_dir, run_dir, "meta.yaml")
        with open(meta_file_path) as meta_file:
            if yaml.safe_load(meta_file.read()) is None:
                print("Run data in file %s was malformed" % meta_file_path)
                if delete:
                    do_delete = True
                else:
                    do_delete = (
                        input("Do you want to delete this run? y to delete") == "y"
                    )
                if do_delete:
                    shutil.rmtree(str(Path(meta_file_path).parent))
                    print("Deleted")
                else:
                    print("Not deleted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to help identify and delete corrupted runs in mlflow."
    )
    parser.add_argument(
        "filestore_root_dir", type=str, help="Path to the root directory for mlflow"
    )
    parser.add_argument(
        "experiment_id",
        type=int,
        help="ID of the problematic experiment. Can be found in the url in mlflow ui",
    )
    parser.add_argument(
        "-d",
        action="store_true",
        help="Do not ask for permission to delete, but do it automatically",
    )

    args = parser.parse_args()
    identify_corrupted(args.filestore_root_dir, args.experiment_id, delete=args.d)
