# cosutils

COS/S3 utility component providing common object-storage operations.

## CLI

```bash
claimed run claimed.components.util.cosutils --help
```

```bash
claimed run claimed.components.util.cosutils \
    --cos-connection s3://KEY:SECRET@endpoint/bucket/path \
    --operation <op> \
    --local-path <local_path> \
    [--recursive true] \
    [--log-level DEBUG]
```

## Operations

| `--operation` | Description |
|---|---|
| `ls` | List objects at the path |
| `find` | Recursively find all objects |
| `mkdir` | Create a bucket/prefix |
| `get` | Download object(s) to `local_path` |
| `put` | Upload `local_path` to the COS path |
| `rm` | Delete object(s) |
| `glob` | Return all paths matching a glob pattern |
| `sync_to_cos` | Upload only changed local files to COS |
| `sync_to_local` | Download only changed COS objects to local |

## Examples

```bash
# List a bucket
claimed run claimed.components.util.cosutils \
    --cos-connection "s3://KEY:SECRET@s3.eu-de.cloud-object-storage.appdomain.cloud/my-bucket" \
    --operation ls \
    --local-path .

# Download a single file
claimed run claimed.components.util.cosutils \
    --cos-connection "s3://KEY:SECRET@s3.eu-de.cloud-object-storage.appdomain.cloud/my-bucket/model.zip" \
    --operation get \
    --local-path .

# Upload an entire directory
claimed run claimed.components.util.cosutils \
    --cos-connection "s3://KEY:SECRET@s3.eu-de.cloud-object-storage.appdomain.cloud/my-bucket/output/" \
    --operation put \
    --local-path ./results \
    --recursive true
```

## Python API

::: claimed.components.util.cosutils
    options:
      members:
        - run
