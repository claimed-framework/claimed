import json
import boto3
import jsonschema
from jsonschema import validate
import os

class COSKVStore:
    def __init__(self, bucket_name, schema, cos_client=None):
        """
        Initialize the COS Key-Value store.
        :param bucket_name: Name of the COS bucket.
        :param schema: JSON Schema to validate values.
        :param cos_client: Optional COS client instance (for dependency injection).
        """
        self.bucket_name = bucket_name
        self.schema = schema
        self.cos_client = cos_client or boto3.client('s3')

    def put(self, key, value):
        """
        Store a value in COS after validating against the JSON schema.
        :param key: The key under which the value is stored.
        :param value: The value to store (must be JSON-serializable).
        """
        try:
            validate(instance=value, schema=self.schema)
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Validation error: {e.message}")
        self.cos_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json.dumps(value)
        )

    def get(self, key):
        """
        Retrieve a value from COS.
        :param key: The key to retrieve.
        :return: The stored value as a dictionary.
        """
        try:
            response = self.cos_client.get_object(Bucket=self.bucket_name, Key=key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except self.cos_client.exceptions.NoSuchKey:
            raise KeyError(f"Key '{key}' not found in bucket '{self.bucket_name}'")

    def delete(self, key):
        """
        Delete a key-value pair from COS.
        :param key: The key to delete.
        """
        self.cos_client.delete_object(Bucket=self.bucket_name, Key=key)

    def list_keys(self):
        """
        List all keys in the COS bucket.
        :return: A list of keys.
        """
        response = self.cos_client.list_objects_v2(Bucket=self.bucket_name)
        return [obj['Key'] for obj in response.get('Contents', [])]

def load_schemas(schema_folder):
    """
    Loads all JSON schemas from the given folder.
    :param schema_folder: Path to the folder containing JSON schema files.
    :return: A dictionary of schema names and their corresponding JSON objects.
    """
    schemas = {}
    for filename in os.listdir(schema_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(schema_folder, filename)
            with open(filepath, 'r') as f:
                schemas[filename[:-5]] = json.load(f)
    return schemas

# Example Usage
if __name__ == "__main__":
    schema_folder = "../schemas"  
    schemas = load_schemas(schema_folder)

    if "example_schema" in schemas:
        example_schema = schemas["example_schema"]
        cos_store = COSKVStore("my-cos-bucket", example_schema)

        test_data = {
            "id": "model_123",
            "name": "MyModel",
            "framework": "TensorFlow",
            "version": "2.10",
            "description": "A simple neural network model.",
            "metrics": {
                "accuracy": 0.95,
                "loss": 0.1
            }
        }
        cos_store.put("model_123", test_data)
        print(cos_store.get("model_123"))

        #Example data without the metric key, which is not required
        test_data_no_metrics = {
            "id": "model_456",
            "name": "MyOtherModel",
            "framework": "PyTorch",
            "version": "1.12",
            "description": "Another neural network model.",
        }
        cos_store.put("model_456", test_data_no_metrics)
        print(cos_store.get("model_456"))

    else:
        print("Schema 'example_schema' not found.")