"""
Logistic Regression Training Operator

This operator trains a logistic regression model using data specified in a configuration file.
It supports distributed training via a local Dask cluster for efficient parallel computation.
The config should define paths to training data, output model location, and training parameters.

The Dask cluster is started locally with 4 workers, each using 1 thread and 12 GiB of memory.

Configuration is required via `--config` flag or `config` environment variable.

"""

# operators/create_training_zarr.py
import os, argparse, dask.config
from dask.distributed import LocalCluster, Client
from dataset_generation.training_ml_wf.training_logistic import train_logistic


def parse_args() -> str:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.environ.get("config"))
    a = p.parse_args()
    if a.config is None:
        p.error("--config missing (and env var 'config' not set)")
    return a.config


def main() -> None:
    cfg = parse_args()
    dask.config.set(
        {
            "dataframe.shuffle.method": "p2p",
            "distributed.worker.memory.target": 0.6,
            "distributed.worker.memory.spill": 0.7,
            "distributed.worker.memory.pause": 0.8,
            "distributed.worker.memory.terminate": 0.95,
        }
    )
    with (
        LocalCluster(
            n_workers=4,
            threads_per_worker=1,
            memory_limit="12GiB",
            local_directory="/tmp/dask-worker-space",
        ) as cluster,
        Client(cluster),
    ):
        train_logistic(cfg)


if __name__ == "__main__":
    main()
