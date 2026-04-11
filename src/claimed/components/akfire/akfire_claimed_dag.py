from __future__ import annotations

import os, shutil
from datetime import timedelta
from pathlib import Path
import pendulum

from airflow.decorators import dag, task
from airflow.models.baseoperator import chain
from airflow.operators.bash import BashOperator

from dataset_generation.utilities.s3_utilities import get_s3_fs


# Environment variables (loaded once at DAG start)
AKFIRE_ACCESS_KEY = os.getenv("AKFIRE_ACCESS_KEY")
AKFIRE_SECRET_KEY = os.getenv("AKFIRE_SECRET_KEY")
AKFIRE_BUCKET = os.getenv("AKFIRE_BUCKET")
AKFIRE_ENDPOINT = os.getenv("AKFIRE_ENDPOINT")

THIS_DIR = Path(__file__).resolve().parent
COMPONENTS_DIR = THIS_DIR
VERSION = "0.1.0"


def op_path(name: str) -> str:
    """returns the claimed-operators/<name>:<VERSION> folder"""
    return str(COMPONENTS_DIR / "claimed-operators")


ENV_VARS: dict[str, str] = {
    "AKFIRE_ACCESS_KEY": AKFIRE_ACCESS_KEY,
    "AKFIRE_SECRET_KEY": AKFIRE_SECRET_KEY,
    "AKFIRE_BUCKET": AKFIRE_BUCKET,
    "AKFIRE_ENDPOINT": AKFIRE_ENDPOINT,
    "CLAIMED_COMPONENTS_DIR": str(COMPONENTS_DIR),
}

START_DATE = pendulum.datetime(2024, 6, 5, 8, 0, tz="Europe/Madrid")
DAG_ID = "claimed_pipeline_dataset_generation"
TAGS = ["claimed", "dataset-generation", "fire"]
CATCHUP = False

DEFAULT_ARGS = {
    "env": {
        **ENV_VARS,
        "SKIP_TASKS": "",
    },
    "retries": 10,
    "retry_delay": timedelta(minutes=2),
}


def claimed_cmd(component: str) -> str:
    """injects the env var «config» and executes the component"""
    return f"/home/airflow/.local/bin/claimed --component containerless/{component}:{VERSION}"


@dag(
    dag_id=DAG_ID,
    schedule_interval=None,
    start_date=START_DATE,
    catchup=CATCHUP,
    tags=TAGS,
    default_args=DEFAULT_ARGS,
)
def claimed_dataset_pipeline():
    tmp_base = THIS_DIR / Path("tmp/akfire_dag")
    config_file = tmp_base / "config.json"

    @task
    def download_config() -> str:
        fs = get_s3_fs()
        tmp_base.mkdir(parents=True, exist_ok=True)
        fs.get(f"{ENV_VARS['AKFIRE_BUCKET']}/config.json", str(config_file))
        return str(config_file)

    @task(trigger_rule="all_done")
    def cleanup(path: str):
        """Remove the temporary directory and its contents."""
        if tmp_base.exists():
            shutil.rmtree(tmp_base)

    # ── compact definition of the six steps ──────────────────────────
    operators = {
        "create_training_zarr": "create_training_zarr",
        "train_logistic": "train_logistic",
        "logistic_prediction": "logistic_prediction",
        "optimize_xgb": "optimize_xgb_hyperparameters_from_df",
        "train_xgboost": "training_xgboost",
        "xgboost_prediction": "xgboost_prediction",
    }

    SKIP_TASKS = DEFAULT_ARGS["env"]["SKIP_TASKS"].split(",")

    tasks = {}
    for task_id, comp_name in operators.items():
        if task_id in SKIP_TASKS:
            tasks[task_id] = BashOperator(
                task_id=task_id,
                bash_command="echo 'Skipped by config'",
            )
        else:
            tasks[task_id] = BashOperator(
                task_id=task_id,
                bash_command=claimed_cmd(comp_name),
                env={  # inherits ENV_VARS + individual component path
                    **ENV_VARS,
                    "CLAIMED_CONTAINERLESS_OPERATOR_PATH": op_path(comp_name),
                    "config": f"{config_file}",
                },
            )

    cfg = download_config()
    chain(
        cfg,
        tasks["create_training_zarr"],
        tasks["train_logistic"],
        tasks["logistic_prediction"],
        tasks["optimize_xgb"],
        tasks["train_xgboost"],
        tasks["xgboost_prediction"],
        cleanup(cfg),
    )


if __name__ == "__main__":
    claimed_dataset_pipeline().test()


claimed_dataset_pipeline()
