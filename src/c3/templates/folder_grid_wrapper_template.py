"""
${component_name} got wrapped by folder_grid_wrapper, which wraps any CLAIMED component and implements folder-level locking.
This folder grid wrapper scans immediate subdirectories of sgw_source_folder and for each folder the ${component_process} function is called once.
Locking is achieved by creating files in the target directory using the pattern <folder>.{STATUS} where STATUS in:
LOCKED
PROCESSED
FAILED


CLAIMED component description: ${component_description}
"""

# pip install pandas

# component dependencies
# ${component_dependencies}

import os
import json
import random
import logging
from pathlib import Path
import pandas as pd

# import component code
from ${component_name} import *

# folder containing input data in single files or subfolders
sgw_source_folder = os.environ.get('sgw_source_folder')

# folder to store the output markers and results
# Default: sgw_source_folder. If equal, entries containing LOCKED or PROCESSED or FAILED are ignored.
sgw_target_folder = os.environ.get('sgw_target_folder', sgw_source_folder)

# component interface
${component_interface}

def _marker_paths(entry_name: str, is_dir: bool):
    """Return (LOCKED, PROCESSED, FAILED) marker paths for a file or a folder."""
    tgt = Path(sgw_target_folder)
    if is_dir:
        # folder markers are directories
        return (
            tgt / f"{entry_name}.LOCKED",
            tgt / f"{entry_name}.PROCESSED",
            tgt / f"{entry_name}.FAILED",
        )
    # file markers are files
    base, ext = os.path.splitext(entry_name)
    return (
        tgt / f"{base}.LOCKED{ext}",
        tgt / f"{base}.PROCESSED{ext}",
        tgt / f"{base}.FAILED{ext}",
    )

def _claimed_any(locked, processed, failed) -> bool:
    return locked.exists() or processed.exists() or failed.exists()

def get_next_batch():
    """Pick a random unclaimed entry from source, supporting files and folders."""
    filtered = []
    with os.scandir(sgw_source_folder) as it:
        for e in it:
            name = e.name

            # If source and target are the same, skip marker entries
            if sgw_source_folder == sgw_target_folder and (
                "LOCKED" in name or "PROCESSED" in name or "FAILED" in name
            ):
                continue

            locked, processed, failed = _marker_paths(name, e.is_dir())
            if not _claimed_any(locked, processed, failed):
                filtered.append((name, e.is_dir()))

    if filtered:
        return random.choice(filtered)  # (name, is_dir)
    return None

def _try_acquire_lock(name: str, is_dir: bool):
    """Create the LOCKED marker atomically and return its Path, or None if already claimed."""
    locked, _, _ = _marker_paths(name, is_dir)
    try:
        if is_dir:
            # atomic directory creation is a good folder lock
            locked.mkdir()
        else:
            # atomic file creation
            fd = os.open(str(locked), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        return locked
    except FileExistsError:
        return None

def process_wrapper(sub_process):
    sgw_target_folder_path = Path(sgw_target_folder)
    sgw_target_folder_path.mkdir(exist_ok=True, parents=True)

    while True:
        nxt = get_next_batch()
        if nxt is None:
            break

        entry_name, is_dir = nxt
        src_path = str(Path(sgw_source_folder) / entry_name)
        locked, processed, failed = _marker_paths(entry_name, is_dir)
        logging.info(f"Processing: {src_path}")

        # Acquire the lock. If we lose the race, pick another entry.
        lock_path = _try_acquire_lock(entry_name, is_dir)
        if lock_path is None:
            continue

        try:
            # Call user component. For folders, src_path points to the folder.
            # The second argument remains the marker path, same as before.
            sub_process(src_path, str(lock_path))

            # Success marker
            lock_path.rename(processed)

        except Exception as e:
            # Failure marker
            lock_path.rename(failed)
            if is_dir:
                # Put the error message inside the FAILED directory
                errfile = Path(failed) / "error.txt"
                errfile.write_text(f"Exception occurred: {str(e)}\n", encoding="utf-8")
            else:
                # For files, FAILED is itself a file; overwrite with the error text
                Path(failed).write_text(f"Exception occurred: {str(e)}\n", encoding="utf-8")
            logging.error(f"Processing failed for {src_path}: {str(e)}")

    logging.info("Finished processing all batches.")

if __name__ == '__main__':
    process_wrapper(${component_process})