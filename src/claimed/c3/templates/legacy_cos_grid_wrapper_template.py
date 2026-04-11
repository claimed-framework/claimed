"""
${component_name} got wrapped by cos_grid_wrapper, which wraps any CLAIMED component and implements the generic grid computing pattern for cos files https://romeokienzler.medium.com/the-generic-grid-computing-pattern-transforms-any-sequential-workflow-step-into-a-transient-grid-c7f3ca7459c8

CLAIMED component description: ${component_description}
"""

# pip install s3fs pandas
# component dependencies
# ${component_dependencies}

import os
import json
import random
import logging
import shutil
import time
import glob
import s3fs
from datetime import datetime
from pathlib import Path
import pandas as pd


# import component code
from ${component_name} import *


def explode_connection_string(cs):
    if cs is None:
        return None, None, None, None
    elif cs.startswith('cos') or cs.startswith('s3'):
        buffer=cs.split('://', 1)[1]
        access_key_id=buffer.split('@')[0].split(':')[0]
        secret_access_key=buffer.split('@')[0].split(':')[1]
        endpoint = f"https://{buffer.split('@')[1].split('/')[0]}"
        path=buffer.split('@')[1].split('/', 1)[1]
        return (access_key_id, secret_access_key, endpoint, path)
    else:
        return (None, None, None, cs)
        # TODO consider cs as secret and grab connection string from kubernetes


# File containing batches. Provided as a comma-separated list of strings or keys in a json dict. All batch file names must contain the batch name.
gw_batch_file = os.environ.get('gw_batch_file', None)
(gw_batch_file_access_key_id, gw_batch_file_secret_access_key, gw_batch_file_endpoint, gw_batch_file) = explode_connection_string(gw_batch_file)
# Optional column name for a csv batch file (default: 'filename')
gw_batch_file_col_name = os.environ.get('gw_batch_file_col_name', 'filename')
# file path pattern like your/path/**/*.tif. Multiple patterns can be separated with commas. It is ignored if gw_batch_file is provided.
gw_file_path_pattern = os.environ.get('gw_file_path_pattern', None)
# pattern for grouping file paths into batches like ".split('.')[-2]". It is ignored if gw_batch_file is provided.
gw_group_by = os.environ.get('gw_group_by', None)

# comma-separated list of additional cos files to copy
gw_additional_source_files = os.environ.get('gw_additional_source_files', '')
# download source cos files to local input path
gw_local_input_path = os.environ.get('gw_local_input_path', 'input')
# upload local target files to target cos path
gw_local_target_path = os.environ.get('gw_local_target_path', 'target')

# cos gw_source_connection
gw_source_connection = os.environ.get('gw_source_connection')
(gw_source_access_key_id, gw_source_secret_access_key, gw_source_endpoint, gw_source_path) = explode_connection_string(gw_source_connection)

# cos gw_target_connection
gw_target_connection = os.environ.get('gw_target_connection')
(gw_target_access_key_id, gw_target_secret_access_key, gw_target_endpoint, gw_target_path) = explode_connection_string(gw_target_connection)

# cos gw_coordinator_connection
gw_coordinator_connection = os.environ.get('gw_coordinator_connection')
(gw_coordinator_access_key_id, gw_coordinator_secret_access_key, gw_coordinator_endpoint, gw_coordinator_path) = explode_connection_string(gw_coordinator_connection)

# lock file suffix
gw_lock_file_suffix = os.environ.get('gw_lock_file_suffix', '.lock')
# processed file suffix
gw_processed_file_suffix = os.environ.get('gw_lock_file_suffix', '.processed')
# error file suffix
gw_error_file_suffix = os.environ.get('gw_error_file_suffix', '.err')
# timeout in seconds to remove lock file from struggling job (default 3 hours)
gw_lock_timeout = int(os.environ.get('gw_lock_timeout', 10800))
# ignore error files and rerun batches with errors
gw_ignore_error_files = bool(os.environ.get('gw_ignore_error_files', False))
# maximal wait time for staggering start
gw_max_time_wait_staggering = int(os.environ.get('gw_max_time_wait_staggering', 60))


# component interface
${component_interface}

# init s3
s3source = s3fs.S3FileSystem(
    anon=False,
    key=gw_source_access_key_id,
    secret=gw_source_secret_access_key,
    client_kwargs={'endpoint_url': gw_source_endpoint})

gw_source_path = Path(gw_source_path)

if gw_target_connection is not None:
    s3target = s3fs.S3FileSystem(
        anon=False,
        key=gw_target_access_key_id,
        secret=gw_target_secret_access_key,
        client_kwargs={'endpoint_url': gw_target_endpoint})
    gw_target_path = Path(gw_target_path)
else:
    logging.debug('Using source path as target path.')
    gw_target_path = gw_source_path
    s3target = s3source

if gw_coordinator_connection is not None:
    s3coordinator = s3fs.S3FileSystem(
        anon=False,
        key=gw_coordinator_access_key_id,
        secret=gw_coordinator_secret_access_key,
        client_kwargs={'endpoint_url': gw_coordinator_endpoint})
    gw_coordinator_path = Path(gw_coordinator_path)
else:
    logging.debug('Using source bucket as coordinator bucket.')
    gw_coordinator_path = gw_source_path
    s3coordinator = s3source

if gw_batch_file_access_key_id is not None:
    s3batch_file = s3fs.S3FileSystem(
        anon=False,
        key=gw_batch_file_access_key_id,
        secret=gw_batch_file_secret_access_key,
        client_kwargs={'endpoint_url': gw_batch_file_endpoint})
else:
    logging.debug('Loading batch file from source s3.')
    s3batch_file = s3source
    gw_batch_file = str(gw_source_path / gw_batch_file)


def load_batches_from_file(batch_file):
    if batch_file.endswith('.json'):
        # load batches from keys of a json file
        logging.info(f'Loading batches from json file: {batch_file}')
        with open(batch_file, 'r') as f:
            batch_dict = json.load(f)
        batches = batch_dict.keys()

    elif batch_file.endswith('.csv'):
        # load batches from keys of a csv file
        logging.info(f'Loading batches from csv file: {batch_file}')
        df = pd.read_csv(batch_file, header='infer')
        assert gw_batch_file_col_name in df.columns, \
            f'gw_batch_file_col_name {gw_batch_file_col_name} not in columns of batch file {batch_file}'
        batches = df[gw_batch_file_col_name].to_list()

    elif batch_file.endswith('.txt'):
        # Load batches from comma-separated txt file
        logging.info(f'Loading comma-separated batch strings from file: {batch_file}')
        with open(batch_file, 'r') as f:
            batch_string = f.read()
        batches = [b.strip() for b in batch_string.split(',')]
    else:
        raise ValueError(f'C3 only supports batch files of type '
                         f'json (batches = dict keys), '
                         f'csv (batches = column values), or '
                         f'txt (batches = comma-seperated list).')

    logging.info(f'Loaded {len(batches)} batches')
    logging.debug(f'List of batches: {batches}')
    assert len(batches) > 0, f"batch_file {batch_file} has no batches."
    return batches


def get_files_from_pattern(file_path_patterns):
    logging.info(f'Start identifying files')
    all_files = []

    # Iterate over comma-separated paths
    for file_path_pattern in file_path_patterns.split(','):
        logging.info(f'Get file paths from pattern: {file_path_pattern}')
        files = s3source.glob(str(gw_source_path / file_path_pattern.strip()))
        if len(files) == 0:
            logging.warning(f"Found no files with file_path_pattern {file_path_pattern}.")
        all_files.extend(files)
    logging.info(f'Found {len(all_files)} cos files')
    return all_files

def identify_batches_from_pattern(file_path_patterns, group_by):
    logging.info(f'Start identifying files and batches')
    batches = set()
    all_files = get_files_from_pattern(file_path_patterns)

    # get batches by applying the group by function to all file paths
    for path_string in all_files:
        part = eval('str(path_string)' + group_by, {"group_by": group_by, "path_string": path_string})
        assert part != '', f'Could not extract batch with path_string {path_string} and group_by {group_by}'
        batches.add(part)

    logging.info(f'Identified {len(batches)} batches')
    logging.debug(f'List of batches: {batches}')

    return batches, all_files


def perform_process(process, batch, cos_files):
    logging.debug(f'Check coordinator files for batch {batch}.')
    # init coordinator files
    coordinator_dir = gw_coordinator_path
    lock_file = str(coordinator_dir / (batch + gw_lock_file_suffix))
    processed_file = str(coordinator_dir / (batch + gw_processed_file_suffix))
    error_file = str(coordinator_dir / (batch + gw_error_file_suffix))

    if s3coordinator.exists(lock_file):
        # remove strugglers
        last_modified = s3coordinator.info(lock_file)['LastModified']
        if (datetime.now(last_modified.tzinfo) - last_modified).total_seconds() > gw_lock_timeout:
            logging.info(f'Lock file {lock_file} is expired.')
            s3coordinator.rm(lock_file)
        else:
            logging.debug(f'Batch {batch} is locked.')
            return

    if s3coordinator.exists(processed_file):
        logging.debug(f'Batch {batch} is processed.')
        return

    if s3coordinator.exists(error_file):
        if gw_ignore_error_files:
            logging.info(f'Ignoring previous error in batch {batch} and rerun.')
        else:
            logging.debug(f'Batch {batch} has error.')
            return

    logging.debug(f'Locking batch {batch}.')
    s3coordinator.touch(lock_file)
    logging.info(f'Processing batch {batch}.')

    # Create input and target directories
    input_path = Path(gw_local_input_path)
    target_path = Path(gw_local_target_path)
    assert not input_path.exists(), (f'gw_local_input_path ({gw_local_input_path}) already exists. '
                                     f'Please provide a new input path.')
    assert not target_path.exists(), (f'gw_local_target_path ({gw_local_target_path}) already exists. '
                                     f'Please provide a new target path.')
    input_path.mkdir(parents=True)
    target_path.mkdir(parents=True)

    # Download cos files to local input folder
    batch_fileset = list(filter(lambda file: batch in file, cos_files))
    if gw_additional_source_files != '':
        additional_source_files = [f.strip() for f in gw_additional_source_files.split(',')]
        batch_fileset.extend(additional_source_files)
    logging.info(f'Downloading {len(batch_fileset)} files from COS')
    for cos_file in batch_fileset:
        local_file = str(input_path / cos_file.split('/', 1)[-1])
        logging.debug(f'Downloading {cos_file} to {local_file}')
        s3source.get(cos_file, local_file)

    # processing files with custom process
    try:
        target_files = process(batch, ${component_inputs})
    except Exception as err:
        logging.exception(err)
        # Write error to file
        with s3coordinator.open(error_file, 'w') as f:
            f.write(f"{type(err).__name__} in batch {batch}: {err}")
        s3coordinator.rm(lock_file)
        logging.error(f'Continue processing.')
        return

    # optional verify target files
    if target_files is not None:
        if isinstance(target_files, str):
            target_files = [target_files]
        for target_file in target_files:
            if not os.path.exists(target_file):
                logging.error(f'Target file {target_file} does not exist for batch {batch}.')
        if any([not str(t).startswith(gw_local_target_path) for t in target_files]):
            logging.warning('Some target files are not in target path. Only files in target path are uploaded.')
    else:
        logging.info(f'Cannot verify batch {batch} (target files not provided). Using files in target_path.')

    # upload files in target path
    local_target_files = list(target_path.glob('*'))
    logging.info(f'Uploading {len(local_target_files)} target files to COS.')
    for local_file in local_target_files:
        cos_file = gw_target_path / local_file.relative_to(target_path)
        logging.debug(f'Uploading {local_file} to {cos_file}')
        s3target.put(str(local_file), str(cos_file))

    logging.info(f'Remove local input and target files.')
    shutil.rmtree(input_path)
    shutil.rmtree(target_path)

    logging.info(f'Finished Batch {batch}.')
    s3coordinator.touch(processed_file)
    # Remove lock file
    if s3coordinator.exists(lock_file):
        s3coordinator.rm(lock_file)
    else:
        logging.warning(f'Lock file {lock_file} was removed by another process. '
                        f'Consider increasing gw_lock_timeout (currently {gw_lock_timeout}s) to repeated processing.')


def process_wrapper(sub_process):
    delay = random.randint(0, gw_max_time_wait_staggering)
    logging.info(f'Staggering start, waiting for {delay} seconds')
    time.sleep(delay)

    # Init coordinator dir
    coordinator_dir =  gw_coordinator_path
    s3coordinator.makedirs(coordinator_dir, exist_ok=True)

    # get batches
    cos_gw_batch_file = str(gw_source_path / gw_batch_file)
    if (gw_batch_file is not None and (os.path.isfile(gw_batch_file) or s3source.exists(cos_gw_batch_file))):
        if not os.path.isfile(gw_batch_file):
            # Download batch file from s3
            if s3batch_file.exists(gw_batch_file):
                s3batch_file.get(gw_batch_file, gw_batch_file)
            else:
                s3batch_file.get(str(gw_source_path / gw_batch_file), gw_batch_file)
        batches = load_batches_from_file(gw_batch_file)
        if gw_file_path_pattern:
            cos_files = get_files_from_pattern(gw_file_path_pattern)
        else:
            logging.warning('gw_file_path_pattern is not provided. '
                            'Grid wrapper expects the wrapped operator to handle COS files instead of the automatic download and upload.')
            cos_files = []
    elif gw_file_path_pattern is not None and gw_group_by is not None:
        batches, cos_files = identify_batches_from_pattern(gw_file_path_pattern, gw_group_by)
    else:
        raise ValueError("Cannot identify batches. "
                         "Provide valid gw_batch_file (local path or path within source bucket) "
                         "or gw_file_path_pattern and gw_group_by.")

    # Iterate over all batches
    for batch in batches:
        perform_process(sub_process, batch, cos_files)

    # Check and log status of batches
    processed_status = [s3coordinator.exists(coordinator_dir / (batch + gw_processed_file_suffix)) for batch in batches]
    lock_status = [s3coordinator.exists(coordinator_dir / (batch + gw_lock_file_suffix)) for batch in batches]
    error_status = [s3coordinator.exists(coordinator_dir / (batch + gw_error_file_suffix)) for batch in batches]

    logging.info(f'Finished current process. Status batches: '
                 f'{sum(processed_status)} processed / {sum(lock_status)} locked / {sum(error_status)} errors / {len(processed_status)} total')

    if sum(error_status):
        logging.error(f'Found errors! Resolve errors and rerun operator with gw_ignore_error_files=True.')
        # print all error messages
        for error_file in s3coordinator.glob(str(coordinator_dir / ('**/*' + gw_error_file_suffix))):
            with s3coordinator.open(error_file, 'r') as f:
                logging.error(f.read())


if __name__ == '__main__':
    process_wrapper(${component_process})
