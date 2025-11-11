"""
component_simple_grid_wrapper_example got wrapped by grid_wrapper, which wraps any CLAIMED component and implements the generic grid computing pattern https://romeokienzler.medium.com/the-generic-grid-computing-pattern-transforms-any-sequential-workflow-step-into-a-transient-grid-c7f3ca7459c8
This simple grid wrapper just scans a folder and for each file the grid_process function is called. Locking is achieved the following way:
Given source file1.ext is processed, simple_grid_wrapper creates files in the target_directory following the pattern file1.{STATUS}.ext where STATUS in:
LOCKED
PROCESSED
FAILED


CLAIMED component description: component-simple-grid-wrapper-example
"""

# pip install pandas

# component dependencies
# 

import os
import json
import random
import logging
import time
import glob
from pathlib import Path
import pandas as pd

# import component code
from component_simple_grid_wrapper_example import *


#folder containing input data in single files
sgw_source_folder = os.environ.get('sgw_source_folder')

#folder to store the output data in single files. Default: sgw_source_folder, in case sgw_source_folder==sgw_target_folder, files containing .LOCKED., .PROCESSED., .FAILED. are ignored
sgw_target_folder = os.environ.get('sgw_target_folder', sgw_source_folder)

# component interface


def get_next_batch():
    files = os.listdir(sgw_source_folder)
    if sgw_source_folder == sgw_target_folder:
        files = [
            f for f in files
            if not any(keyword in f for keyword in ["LOCKED", "PROCESSED", "FAILED"])
        ]

    # Filter files and check if corresponding target file exists
    filtered_files = []
    for file in files:
        file_name, file_ext = os.path.splitext(file)

        # Create target file names with LOCKED, PROCESSED, FAILED extensions
        target_file_locked = f"{file_name}.LOCKED{file_ext}"
        target_file_processed = f"{file_name}.PROCESSED{file_ext}"
        target_file_failed = f"{file_name}.FAILED{file_ext}"

        # Check if any of the target files exists
        if not any(
                os.path.exists(os.path.join(sgw_target_folder, target_file))
                for target_file in [target_file_locked, target_file_processed, target_file_failed]
        ):
            filtered_files.append(file)

    if filtered_files:
        return random.choice(filtered_files)
    else:
        return None


def process_wrapper(sub_process):
    sgw_target_folder_path = Path(sgw_target_folder)
    sgw_target_folder_path.mkdir(exist_ok=True, parents=True)

    while True:
        file_to_process = get_next_batch()
        logging.info(f"Processing batch: {file_to_process}")
        if file_to_process is None:
            break

        file_name = Path(file_to_process).stem
        file_ext = Path(file_to_process).suffix
        locked_file = sgw_target_folder+f"/{file_name}.LOCKED{file_ext}"
        locked_file_path = Path(locked_file)

        try:
            locked_file_path.touch()
            sub_process(sgw_source_folder +'/'+ file_to_process, locked_file)
            processed_file = sgw_target_folder+f"/{file_name}.PROCESSED{file_ext}"
            locked_file_path.rename(processed_file)

        except Exception as e:
            failed_file = sgw_target_folder+f"/{file_name}.FAILED{file_ext}"
            locked_file_path.rename(failed_file)

            with open(failed_file, 'w') as f:
                f.write(f"Exception occurred: {str(e)}\n")

            logging.error(f"Processing failed for {file_to_process}: {str(e)}")

    logging.info("Finished processing all batches.")


if __name__ == '__main__':
    process_wrapper(grid_process) 
