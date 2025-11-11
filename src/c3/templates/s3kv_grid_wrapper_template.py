"""
${component_name} got wrapped by grid_wrapper, which wraps any CLAIMED component and implements the generic grid computing pattern https://romeokienzler.medium.com/the-generic-grid-computing-pattern-transforms-any-sequential-workflow-step-into-a-transient-grid-c7f3ca7459c8

CLAIMED component description: ${component_description}
"""

# pip install s3fs boto3 pandas
# component dependencies
# ${component_dependencies}

import os
import json
import random
import logging
import time
import glob
from pathlib import Path
import pandas as pd
import s3fs
from hashlib import sha256



# import component code
from ${component_name} import *

#------------------REMOVE once pip install for s3kv is fixed
import os
import time
from datetime import datetime
import shutil
import boto3
import json


class S3KV:
    def __init__(self, s3_endpoint_url:str, bucket_name: str, 
                 aws_access_key_id: str = None, aws_secret_access_key: str = None , enable_local_cache=True):
        """
        Initializes the S3KV object with the given S3 bucket, AWS credentials, and Elasticsearch host.

        :param s3_endpoint_url: The s3 endpoint.
        :param bucket_name: The name of the S3 bucket to use for storing the key-value data.
        :param aws_access_key_id: (Optional) AWS access key ID.
        :param aws_secret_access_key: (Optional) AWS secret access key.
        """
        self.bucket_name = bucket_name
        self.enable_local_cache = enable_local_cache
        self.s3_client = boto3.client(
            's3',
            endpoint_url=s3_endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        if not os.path.exists('/tmp/s3kv_cache'):
            os.makedirs('/tmp/s3kv_cache')

    def _get_object_key(self, key: str) -> str:
        """
        Constructs the S3 object key for the given key.

        :param key: The key used to access the value in the S3 bucket.
        :return: The S3 object key for the given key.
        """
        return f"s3kv/{key}.json"

    def cache_all_keys(self):
        """
        Saves all keys to the local /tmp directory as they are being added.
        """
        keys = self.list_keys()
        for key in keys:
            value = self.get(key)
            if value is not None:
                with open(f'/tmp/s3kv_cache/{key}.json', 'w') as f:
                    json.dump(value, f)

    def get_from_cache(self, key: str) -> dict:
        """
        Retrieves a key from the local cache if present, and clears old cache entries.

        :param key: The key to retrieve from the cache.
        :return: The value associated with the given key if present in the cache, else None.
        """
        self.clear_old_cache()
        cache_path = f'/tmp/s3kv_cache/{key}.json'
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        else:
            return None


    def add(self, key: str, value: dict, metadata: dict = None):
        """
        Adds a new key-value pair to the S3KV database, caches it locally, and sends metadata to Elasticsearch.

        :param key: The key to be added.
        :param value: The value corresponding to the key.
        :param metadata: (Optional) Metadata associated with the data (will be sent to Elasticsearch).
        """
        s3_object_key = self._get_object_key(key)
        serialized_value = json.dumps(value)
        self.s3_client.put_object(Bucket=self.bucket_name, Key=s3_object_key, Body=serialized_value)

        with open(f'/tmp/s3kv_cache/{key}.json', 'w') as f:
            json.dump(value, f)



    def delete(self, key: str):
        """
        Deletes a key-value pair from the S3KV database.

        :param key: The key to be deleted.
        """
        s3_object_key = self._get_object_key(key)
        self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_object_key)

        cache_path = f'/tmp/s3kv_cache/{key}.json'
        if os.path.exists(cache_path):
            os.remove(cache_path)


    def get(self, key: str, default: dict = None) -> dict:
        """
        Retrieves the value associated with the given key from the S3KV database.

        :param key: The key whose value is to be retrieved.
        :param default: (Optional) The default value to return if the key does not exist.
        :return: The value associated with the given key, or the default value if the key does not exist.
        """
        s3_object_key = self._get_object_key(key)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_object_key)
            value = response['Body'].read()
            return json.loads(value)
        except self.s3_client.exceptions.NoSuchKey:
            return default


    def list_keys(self) -> list:
        """
        Lists all the keys in the S3KV database.

        :return: A list of all keys in the database.
        """
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix="")
        keys = [obj['Key'][5:-5] for obj in response.get('Contents', []) if obj['Key'].endswith('.json')]
        return keys


    def clear_cache(self):
        """
        Clears the local cache by removing all cached JSON files.
        """
        cache_directory = '/tmp/s3kv_cache'
        if os.path.exists(cache_directory):
            shutil.rmtree(cache_directory)
        os.makedirs('/tmp/s3kv_cache')


    def clear_old_cache(self, max_days: int = 7):
        """
        Clears the cache for keys that have been in the cache for longer than a specific number of days.

        :param max_days: The maximum number of days a key can stay in the cache before being cleared.
        """
        cache_directory = '/tmp/s3kv_cache'
        current_time = time.time()

        if os.path.exists(cache_directory):
            for filename in os.listdir(cache_directory):
                file_path = os.path.join(cache_directory, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_days * 86400:  # Convert days to seconds
                        os.remove(file_path)


    def clear_cache_for_key(self, key: str):
        """
        Clears the local cache for a specific key in the S3KV database.

        :param key: The key for which to clear the local cache.
        """
        cache_path = f'/tmp/s3kv_cache/{key}.json'
        if os.path.exists(cache_path):
            os.remove(cache_path)


    def key_exists(self, key: str) -> bool:
        """
        Checks if a key exists in the S3KV database.

        :param key: The key to check.
        :return: True if the key exists, False otherwise.
        """
        s3_object_key = self._get_object_key(key)
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_object_key)
            return True
        except Exception as e:
                # Return false even if response is unauthorized or similar
                return False


    def list_keys_with_prefix(self, prefix: str) -> list:
        """
        Lists all the keys in the S3KV database that have a specific prefix.

        :param prefix: The prefix to filter the keys.
        :return: A list of keys in the database that have the specified prefix.
        """
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        keys = [obj['Key'][5:-5] for obj in response.get('Contents', []) if obj['Key'].endswith('.json')]
        return keys


    def copy_key(self, source_key: str, destination_key: str):
        """
        Copies the value of one key to another key in the S3KV database.

        :param source_key: The key whose value will be copied.
        :param destination_key: The key to which the value will be copied.
        """
        source_s3_object_key = self._get_object_key(source_key)
        destination_s3_object_key = self._get_object_key(destination_key)

        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=source_s3_object_key)
        value = response['Body'].read()

        self.s3_client.put_object(Bucket=self.bucket_name, Key=destination_s3_object_key, Body=value)

        # Copy the key in the local cache if it exists
        source_cache_path = f'/tmp/s3kv_cache/{source_key}.json'
        destination_cache_path = f'/tmp/s3kv_cache/{destination_key}.json'
        if os.path.exists(source_cache_path):
            shutil.copy(source_cache_path, destination_cache_path)


    def get_key_size(self, key: str) -> int:
        """
        Gets the size (file size) of a key in the S3KV database.

        :param key: The key whose size will be retrieved.
        :return: The size (file size) of the key in bytes, or 0 if the key does not exist.
        """
        s3_object_key = self._get_object_key(key)
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_object_key)
            return response['ContentLength']
        except self.s3_client.exceptions.NoSuchKey:
            return 0


    def get_key_last_updated_time(self, key: str) -> float:
        """
        Gets the last updated time of a key in the S3KV database.

        :param key: The key whose last updated time will be retrieved.
        :return: The last updated time of the key as a floating-point timestamp, or 0 if the key does not exist.
        """
        s3_object_key = self._get_object_key(key)
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_object_key)
            last_modified = response['LastModified']
            st = time.mktime(last_modified.timetuple())

            return  datetime.fromtimestamp(st)

        except self.s3_client.exceptions.NoSuchKey:
            return 0


    def set_bucket_policy(self):
        """
        Sets a bucket policy to grant read and write access to specific keys used by the S3KV library.
        """
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "S3KVReadWriteAccess",
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": "*"
                    },
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject"
                    ],
                    "Resource": f"arn:aws:s3:::{self.bucket_name}/s3kv/*"
                }
            ]
        }

        policy_json = json.dumps(policy)
        self.s3_client.put_bucket_policy(Bucket=self.bucket_name, Policy=policy_json)


    def tag_key(self, key: str, tags: dict):
        """
        Tags a key in the S3KV database with the provided tags.

        :param key: The key to be tagged.
        :param tags: A dictionary containing the tags to be added to the key.
                     For example, {'TagKey1': 'TagValue1', 'TagKey2': 'TagValue2'}
        """
        s3_object_key = self._get_object_key(key)

        # Convert the tags dictionary to a format compatible with the `put_object_tagging` method
        tagging = {'TagSet': [{'Key': k, 'Value': v} for k, v in tags.items()]}

        # Apply the tags to the object
        self.s3_client.put_object_tagging(Bucket=self.bucket_name, Key=s3_object_key, Tagging=tagging)


    def tag_keys_with_prefix(self, prefix: str, tags: dict):
        """
        Tags all keys in the S3KV database with the provided prefix with the specified tags.

        :param prefix: The prefix of the keys to be tagged.
        :param tags: A dictionary containing the tags to be added to the keys.
                     For example, {'TagKey1': 'TagValue1', 'TagKey2': 'TagValue2'}
        """
        keys_to_tag = self.list_keys_with_prefix(prefix)

        for key in keys_to_tag:
            self.tag_key(key, tags)


    def merge_keys(self, source_keys: list, destination_key: str):
        """
        Merges the values of source keys into the value of the destination key in the S3KV database.

        :param source_keys: A list of source keys whose values will be merged.
        :param destination_key: The key whose value will be updated by merging the source values.
        """
        destination_s3_object_key = self._get_object_key(destination_key)

        # Initialize an empty dictionary for the destination value
        destination_value = {}

        # Retrieve and merge values from source keys
        for source_key in source_keys:
            source_value = self.get(source_key)
            if source_value:
                destination_value.update(source_value)

        # Update the destination value in the S3 bucket
        serialized_value = json.dumps(destination_value)
        self.s3_client.put_object(Bucket=self.bucket_name, Key=destination_s3_object_key, Body=serialized_value)

        # Update the value in the local cache if it exists
        destination_cache_path = f'/tmp/s3kv_cache/{destination_key}.json'
        with open(destination_cache_path, 'w') as f:
            json.dump(destination_value, f)



    def find_keys_by_tag_value(self, tag_key: str, tag_value: str) -> list:
        """
        Finds keys in the S3KV database based on the value of a specific tag.

        :param tag_key: The tag key to search for.
        :param tag_value: The tag value to search for.
        :return: A list of keys that have the specified tag key with the specified value.
        """
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix="s3kv/")
        keys_with_tag = []

        for obj in response.get('Contents', []):
            s3_object_key = obj['Key']
            tags = self.get_tags(s3_object_key)
            if tags and tag_key in tags and tags[tag_key] == tag_value:
                keys_with_tag.append(s3_object_key[5:-5])  # Extract the key name

        return keys_with_tag

    def get_tags(self, s3_object_key: str) -> dict:
        """
        Gets the tags of an object in the S3KV database.

        :param s3_object_key: The S3 object key whose tags will be retrieved.
        :return: A dictionary containing the tags of the object.
        """
        response = self.s3_client.get_object_tagging(Bucket=self.bucket_name, Key=s3_object_key)
        tags = {}
        for tag in response.get('TagSet', []):
            tags[tag['Key']] = tag['Value']
        return tags
    


    def place_retention_lock(self, key: str, retention_days: int):
        """
        Places a retention lock on a key in the S3KV database for the specified number of days.

        :param key: The key to place the retention lock on.
        :param retention_days: The number of days to lock the key for.
        """
        s3_object_key = self._get_object_key(key)
        print(s3_object_key)

        retention_period = retention_days * 24 * 60 * 60  # Convert days to seconds

        self.s3_client.put_object_retention(
            Bucket=self.bucket_name,
            Key=s3_object_key,
            Retention={
                'Mode': 'GOVERNANCE',
                'RetainUntilDate': int(time.time()) + retention_period
            }
        )


    def remove_retention_lock(self, key: str):
        """
        Removes the retention lock from a key in the S3KV database.

        :param key: The key to remove the retention lock from.
        """
        s3_object_key = self._get_object_key(key)

        self.s3_client.put_object_retention(
            Bucket=self.bucket_name,
            Key=s3_object_key,
            BypassGovernanceRetention=True,
            Retention={
                
            }
        )


    def delete_by_tag(self, tag_key: str, tag_value: str):
        """
        Deletes keys in the S3KV database based on a specific tag.

        :param tag_key: The tag key to match for deletion.
        :param tag_value: The tag value to match for deletion.
        """
        keys_to_delete = self.find_keys_by_tag_value(tag_key, tag_value)

        for key in keys_to_delete:
            self.delete(key)


    def apply_legal_hold(self, key: str):
        """
        Applies a legal hold on a key in the S3KV database.

        :param key: The key on which to apply the legal hold.
        """
        s3_object_key = self._get_object_key(key)

        self.s3_client.put_object_legal_hold(
            Bucket=self.bucket_name,
            Key=s3_object_key,
            LegalHold={
                'Status': 'ON'
            }
        )





    def is_legal_hold_applied(self, key: str) -> bool:
        """
        Checks if a key in the S3KV database is under legal hold.

        :param key: The key to check for legal hold.
        :return: True if the key is under legal hold, False otherwise.
        """
        s3_object_key = self._get_object_key(key)

        response = self.s3_client.get_object_legal_hold(Bucket=self.bucket_name, Key=s3_object_key)

        legal_hold_status = response.get('LegalHold', {}).get('Status')
        return legal_hold_status == 'ON'


    def release_legal_hold(self, key: str):
        """
        Releases a key from legal hold in the S3KV database.

        :param key: The key to release from legal hold.
        """
        s3_object_key = self._get_object_key(key)

        self.s3_client.put_object_legal_hold(
            Bucket=self.bucket_name,
            Key=s3_object_key,
            LegalHold={
                'Status': 'OFF'
            }
        )

#-----------------------------------------------------------


def explode_connection_string(cs):
    if cs is None:
        return None, None, None, None
    if cs.startswith('cos') or cs.startswith('s3'):
        buffer=cs.split('://')[1]
        access_key_id=buffer.split('@')[0].split(':')[0]
        secret_access_key=buffer.split('@')[0].split(':')[1]
        endpoint=f"https://{buffer.split('@')[1].split('/')[0]}"
        path=buffer.split('@')[1].split('/', 1)[1]
        return (access_key_id, secret_access_key, endpoint, path)
    else:
        return (None, None, None, cs)
        # TODO consider cs as secret and grab connection string from kubernetes



# File with batches. Provided as a comma-separated list of strings,  keys in a json dict or single column CSV with 'filename' has header. Either local path as [cos|s3]://user:pw@endpoint/path
gw_batch_file = os.environ.get('gw_batch_file', None)
(gw_batch_file_access_key_id, gw_batch_file_secret_access_key, gw_batch_file_endpoint, gw_batch_file) = explode_connection_string(gw_batch_file)
# Optional column name for a csv batch file (default: 'filename')
gw_batch_file_col_name = os.environ.get('gw_batch_file_col_name', 'filename')

# cos gw_coordinator_connection
gw_coordinator_connection = os.environ.get('gw_coordinator_connection')
(gw_coordinator_access_key_id, gw_coordinator_secret_access_key, gw_coordinator_endpoint, gw_coordinator_path) = explode_connection_string(gw_coordinator_connection)

# maximal wait time for staggering start
gw_max_time_wait_staggering = int(os.environ.get('gw_max_time_wait_staggering',60))

# component interface
#${component_interface}

def load_batches_from_file(batch_file):
    # Download batch file from s3
    s3_batch_file = s3fs.S3FileSystem(
        anon=False,
        key=gw_batch_file_access_key_id,
        secret=gw_batch_file_secret_access_key,
        client_kwargs={'endpoint_url': gw_batch_file_endpoint})
    s3_batch_file.get(batch_file, batch_file)

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


def perform_process(process, batch, coordinator):
    logging.debug(f'Check coordinator files for batch {batch}.')

    batch_id = sha256(batch.encode('utf-8')).hexdigest() # ensure no special characters break cos
    logging.info(f'Generating {batch_id} for {batch}')

    if coordinator.key_exists(batch_id):
        if coordinator.get(batch_id) == 'locked':
            logging.debug(f'Batch {batch_id} is locked')
            return
        elif coordinator.get(batch_id) == 'processed':
            logging.debug(f'Batch {batch_id} is processed')
            return
        else:
            logging.debug(f'Batch {batch_id} is failed')
            return


    logging.debug(f'Locking batch {batch_id}.')
    coordinator.add(batch_id,'locked')

    # processing files with custom process
    logging.info(f'Processing batch {batch_id}.')
    try:
        process(batch, ${component_inputs})
    except Exception as err:
        logging.exception(err)
        coordinator.add(batch_id,f"{type(err).__name__} in batch {batch_id}: {err}")
        logging.error(f'Continue processing.')
        return

    logging.info(f'Finished Batch {batch_id}.')
    coordinator.add(batch_id,'processed')


def process_wrapper(sub_process):
    delay = random.randint(0, gw_max_time_wait_staggering)
    logging.info(f'Staggering start, waiting for {delay} seconds')
    time.sleep(delay)

    # Init coordinator
    coordinator = S3KV(gw_coordinator_endpoint,
             gw_coordinator_path, 
             gw_coordinator_access_key_id, gw_coordinator_secret_access_key,
             enable_local_cache=False)


    # get batches
    batches = load_batches_from_file(gw_batch_file)

    # Iterate over all batches
    for batch in batches:
        perform_process(sub_process, batch, coordinator)

    # Check and log status of batches
    processed_status = sum(coordinator.get(batch_id) == 'processed' for batch_id in batches)
    lock_status = sum(coordinator.get(batch_id) == 'locked' for batch_id in batches)
    exists_status = sum(coordinator.key_exists(batch_id) for batch_id in batches)
    error_status = exists_status - processed_status - lock_status

    logging.info(f'Finished current process. Status batches: '
                 f'{processed_status} processed / {lock_status} locked / {error_status} errors / {len(batches)} total')


if __name__ == '__main__':
    process_wrapper(${component_process})
