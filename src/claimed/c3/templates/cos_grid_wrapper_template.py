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
# cos gw_coordinator_connection
gw_coordinator_connection = os.environ.get('gw_coordinator_connection')
(gw_coordinator_access_key_id, gw_coordinator_secret_access_key, gw_coordinator_endpoint, gw_coordinator_path) = explode_connection_string(gw_coordinator_connection)
# timeout in seconds to remove lock file from struggling job (default 3 hours)
gw_lock_timeout = int(os.environ.get('gw_lock_timeout', 10800))
# ignore error files and rerun batches with errors
gw_ignore_error_files = bool(os.environ.get('gw_ignore_error_files', False))
# maximal wait time for staggering start
gw_max_time_wait_staggering = int(os.environ.get('gw_max_time_wait_staggering', 60))

# coordinator file suffix
suffix_lock = '.lock'
suffix_processed = '.processed'
suffix_error = '.err'

# component interface
${component_interface}

# Init s3
s3coordinator = s3fs.S3FileSystem(
    anon=False,
    key=gw_coordinator_access_key_id,
    secret=gw_coordinator_secret_access_key,
    client_kwargs={'endpoint_url': gw_coordinator_endpoint})
gw_coordinator_path = Path(gw_coordinator_path)

if gw_batch_file_access_key_id is not None:
    s3batch_file = s3fs.S3FileSystem(
        anon=False,
        key=gw_batch_file_access_key_id,
        secret=gw_batch_file_secret_access_key,
        client_kwargs={'endpoint_url': gw_batch_file_endpoint})
else:
    logging.debug('Loading batch file from source s3.')
    s3batch_file = s3coordinator


def load_batches_from_file(batch_file):
    if batch_file.endswith('.json'):
        # Load batches from keys of a json file
        logging.info(f'Loading batches from json file: {batch_file}')
        with open(batch_file, 'r') as f:
            batch_dict = json.load(f)
        batches = batch_dict.keys()

    elif batch_file.endswith('.csv'):
        # Load batches from keys of a csv file
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


def perform_process(process, batch):
    logging.debug(f'Check coordinator files for batch {batch}.')
    # Init coordinator files
    lock_file = str(gw_coordinator_path / (batch + suffix_lock))
    processed_file = str(gw_coordinator_path / (batch + suffix_processed))
    error_file = str(gw_coordinator_path / (batch + suffix_error))

    if s3coordinator.exists(lock_file):
        # Remove strugglers
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

    # processing files with custom process
    logging.info(f'Processing batch {batch}.')
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

    logging.info(f'Finished Batch {batch}.')
    s3coordinator.touch(processed_file)
    # Remove lock file
    if s3coordinator.exists(lock_file):
        s3coordinator.rm(lock_file)
    else:
        logging.warning(f'Lock file {lock_file} was removed by another process. '
                        f'Consider increasing gw_lock_timeout to avoid repeated processing (currently {gw_lock_timeout}s).')


def process_wrapper(sub_process):
    delay = random.randint(0, gw_max_time_wait_staggering)
    logging.info(f'Staggering start, waiting for {delay} seconds')
    time.sleep(delay)

    # Init coordinator dir
    s3coordinator.makedirs(gw_coordinator_path, exist_ok=True)

    # Download batch file
    if s3batch_file.exists(gw_batch_file):
        s3batch_file.get(gw_batch_file, gw_batch_file)
    if not os.path.isfile(gw_batch_file):
        # Download batch file from s3 coordinator
        cos_gw_batch_file = str(gw_coordinator_path.split([0]) / gw_batch_file)
        if s3batch_file.exists(cos_gw_batch_file):
            s3batch_file.get(gw_batch_file, gw_batch_file)
        else:
            raise ValueError("Cannot identify batches. Provide valid gw_batch_file "
                             "(local path, path within coordinator bucket, or s3 connection to batch file).")

    # Get batches
    batches = load_batches_from_file(gw_batch_file)

    # Iterate over all batches
    for batch in batches:
        perform_process(sub_process, batch)

    # Check and log status of batches
    processed_status = sum(s3coordinator.exists(gw_coordinator_path / (batch + suffix_processed)) for batch in batches)
    lock_status = sum(s3coordinator.exists(gw_coordinator_path / (batch + suffix_lock)) for batch in batches)
    error_status = sum(s3coordinator.exists(gw_coordinator_path / (batch + suffix_error)) for batch in batches)

    logging.info(f'Finished current process. Status batches: '
                 f'{processed_status} processed / {lock_status} locked / {error_status} errors / {len(batches)} total')

    if error_status:
        logging.error(f'Found errors! Resolve errors and rerun operator with gw_ignore_error_files=True.')
        # Print all error messages
        for error_file in s3coordinator.glob(str(gw_coordinator_path / ('**/*' + suffix_error))):
            with s3coordinator.open(error_file, 'r') as f:
                logging.error(f.read())


if __name__ == '__main__':
    process_wrapper(${component_process})
