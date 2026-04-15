#!/usr/bin/env python
# coding: utf-8

# # util-cos

# This component provides COS utility functions (e.g. creating a bucket, listing contents of a bucket)
# 
# Open Issues:
# - [] make sure endpoint starts with https independent of input start is empty, s3 or s3a
# - [] make sure there is a / symbol between bucket and path although not specified

# In[ ]:


#!pip install aiobotocore botocore s3fs claimed-c3

# In[ ]:


import logging
import os
import re
import s3fs
import sys
import glob
from tqdm import tqdm
from claimed.c3.operator_utils import explode_connection_string

MIN_CHUNK_SIZE = 8 * 1024 * 1024   # 8 MiB
MAX_PARTS      = 9500               # S3 hard limit is 10 000; stay safely below


def _upload(s3, local_file, cos_file):
    """Upload a single file to S3/COS with a byte-level progress bar.

    Chunk size is computed dynamically so the number of multipart parts
    never exceeds the S3/COS limit of 10 000.
    """
    # If cos_file is a bucket root or ends with '/', treat it as a directory prefix
    if cos_file == '' or cos_file.endswith('/') or '/' not in cos_file:
        cos_file = cos_file.rstrip('/') + '/' + os.path.basename(local_file)
    size = os.path.getsize(local_file)
    # Ensure chunk size is large enough to stay within the 10 000-part limit
    chunk_size = max(MIN_CHUNK_SIZE, math.ceil(size / MAX_PARTS))
    desc = os.path.basename(local_file)
    with tqdm(total=size, unit='B', unit_scale=True, unit_divisor=1024,
              desc=f'↑ {desc}', leave=True) as pbar:
        with open(local_file, 'rb') as f_in, \
             s3.open(cos_file, 'wb', block_size=chunk_size) as f_out:
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                pbar.update(len(chunk))


def _download(s3, cos_file, local_file):
    """Download a single file from S3/COS with a byte-level progress bar."""
    os.makedirs(os.path.dirname(local_file) or '.', exist_ok=True)
    size = s3.info(cos_file)['size']
    desc = os.path.basename(cos_file)
    with tqdm(total=size, unit='B', unit_scale=True, unit_divisor=1024,
              desc=f'↓ {desc}', leave=True) as pbar:
        with s3.open(cos_file, 'rb') as f_in, open(local_file, 'wb') as f_out:
            while True:
                chunk = f_in.read(CHUNK_SIZE)
                if not chunk:
                    break
                f_out.write(chunk)
                pbar.update(len(chunk))

# In[ ]:



# cos_connection in format: [cos|s3]://access_key_id:secret_access_key@endpoint/bucket/path
cos_connection = os.environ.get('cos_connection')

# local_path for uploads, downloads, sync
local_path = os.environ.get('local_path')

# recursive
recursive = bool(os.environ.get('recursive','False'))

# operation (mkdir|ls|find|get|put|rm|sync_to_cos|sync_to_local|glob)
operation = os.environ.get('operation')

# log level
log_level = os.environ.get('log_level', 'INFO')

# In[ ]:


def run(
    cos_connection: str,
    local_path: str,
    operation: str,
    recursive: bool = False,
    log_level: str = 'INFO',
) -> None:
    """
    Perform a COS/S3 file operation.

    cos_connection: s3://access_key_id:secret_access_key@endpoint/bucket/path
    operation:      one of mkdir | ls | find | get | put | rm | sync_to_cos | sync_to_local | glob
    local_path:     local file or directory used for get / put / sync operations
    recursive:      apply the operation recursively
    log_level:      logging verbosity: DEBUG | INFO | WARNING | ERROR  (default: INFO)
    """
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    (access_key_id, secret_access_key, endpoint, cos_path) = explode_connection_string(cos_connection)
    s3 = s3fs.S3FileSystem(
        anon=False,
        key=access_key_id,
        secret=secret_access_key,
        client_kwargs={'endpoint_url': endpoint}
    )

    if operation == 'mkdir':
        s3.mkdir(cos_path)

    elif operation == 'ls':
        print(s3.ls(cos_path))

    elif operation == 'find':
        print(s3.find(cos_path))

    elif operation == 'put':
        if recursive or os.path.isdir(local_path):
            # gather all files under local_path
            files = [f for f in glob.glob(
                os.path.join(local_path, '**'), recursive=True)
                if os.path.isfile(f)]
            with tqdm(files, unit='file', desc='Uploading') as pbar:
                for f in pbar:
                    rel = os.path.relpath(f, local_path)
                    pbar.set_postfix_str(rel)
                    _upload(s3, f, cos_path.rstrip('/') + '/' + rel)
        else:
            _upload(s3, local_path, cos_path)

    elif operation == 'sync_to_cos':
        files = glob.glob(local_path, recursive=recursive)
        with tqdm(files, unit='file', desc='Syncing → COS') as pbar:
            for file in pbar:
                pbar.set_postfix_str(file)
                logging.info(f'processing {file}')
                if s3.exists(cos_path + file):
                    logging.debug(f's3.info {s3.info(cos_path + file)}')
                    if s3.info(cos_path + file)['size'] != os.path.getsize(file):
                        logging.info(f'uploading {file}')
                        _upload(s3, file, cos_path + file)
                    else:
                        logging.info(f'skipping {file}')
                else:
                    logging.info(f'uploading {file}')
                    _upload(s3, file, cos_path + file)

    elif operation == 'sync_to_local':
        remote_files = [p for p in s3.glob(cos_path)
                        if s3.info(p)['type'] != 'directory']
        with tqdm(remote_files, unit='file', desc='Syncing → local') as pbar:
            for full_path in pbar:
                local_full_path = local_path + full_path
                pbar.set_postfix_str(os.path.basename(full_path))
                logging.info(f'processing {full_path}')
                if os.path.exists(local_full_path):
                    if s3.info(full_path)['size'] != os.path.getsize(local_full_path):
                        logging.info(f'downloading {full_path} to {local_full_path}')
                        _download(s3, full_path, local_full_path)
                    else:
                        logging.info(f'skipping {full_path}')
                else:
                    logging.info(f'downloading {full_path} to {local_full_path}')
                    _download(s3, full_path, local_full_path)

    elif operation == 'get':
        if recursive:
            remote_files = [p for p in s3.find(cos_path)
                            if s3.info(p)['type'] != 'directory']
            with tqdm(remote_files, unit='file', desc='Downloading') as pbar:
                for rp in pbar:
                    rel = rp[len(cos_path):].lstrip('/')
                    lp = os.path.join(local_path, rel)
                    pbar.set_postfix_str(os.path.basename(rp))
                    _download(s3, rp, lp)
        else:
            dest = local_path
            if os.path.isdir(local_path):
                dest = os.path.join(local_path, os.path.basename(cos_path))
            _download(s3, cos_path, dest)

    elif operation == 'rm':
        s3.rm(cos_path, recursive=recursive)

    elif operation == 'glob':
        print(s3.glob(cos_path))

    else:
        logging.error(f'operation unknown: {operation}')

# In[ ]:


if __name__ == "__main__":
    run(cos_connection, local_path, operation, recursive, log_level)

