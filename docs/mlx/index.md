# MLX Backend

The **Machine Learning eXchange (MLX)** backend is responsible for tracking and managing all assets
used and produced by the CLAIMED framework.

---

## What MLX tracks

| Asset type | Description |
|---|---|
| **Datasets** | Input/output data files stored in S3/COS |
| **Models** | Trained model artefacts |
| **Jobs** | Execution records and logs |
| **Pipeline runs** | End-to-end provenance graphs |

---

## Architecture

```
  claimed.c3 grid wrappers
         │
         ▼
  claimed.mlx.s3_kv_store    ← key-value abstraction over S3/COS
         │
         ▼
  claimed.mlx.cos_backend    ← low-level S3/COS operations (s3fs)
         │
         ▼
       S3 / IBM COS
```

---

## Modules

| Module | Description |
|---|---|
| [`cos_backend`](cos-backend.md) | Low-level S3/COS file operations |
| [`s3_kv_store`](s3-kv-store.md) | Key-value store abstraction used by grid wrappers |

---

## Configuration

The MLX backend is configured through connection strings in the standard CLAIMED format:

```
s3://access_key_id:secret_access_key@endpoint_host/bucket/prefix
```

See [operator_utils](../c3/operator-utils.md) for the full connection string specification.
