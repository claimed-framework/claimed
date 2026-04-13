# operator_utils

Shared utility helpers used across C3 and the component library.

## Python API

::: claimed.c3.operator_utils

## Connection String Format

Many CLAIMED components accept a `cos_connection` parameter in the following URI format:

```
[cos|s3]://access_key_id:secret_access_key@endpoint_host/bucket/path
```

**Examples:**

```
s3://AKIAIOSFODNN7EXAMPLE:wJalrXUtnFEMI@s3.us-east-1.amazonaws.com/my-bucket/data/
cos://mykey:mysecret@s3.eu-de.cloud-object-storage.appdomain.cloud/my-bucket/models/
```

### `explode_connection_string(cs)`

Parses the URI into its components:

```python
from claimed.c3.operator_utils import explode_connection_string

access_key_id, secret_access_key, endpoint, path = explode_connection_string(
    's3://KEY:SECRET@s3.eu-de.cloud-object-storage.appdomain.cloud/my-bucket/prefix'
)
# endpoint → 'https://s3.eu-de.cloud-object-storage.appdomain.cloud'
# path     → 'my-bucket/prefix'
```

If the string does not start with `cos://` or `s3://`, the input is returned as-is in the `path` field
(useful when passing a plain local path or a Kubernetes secret reference).
