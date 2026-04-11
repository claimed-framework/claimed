# Wildfire-Risk Pipeline with IBM CLAIMED

A scalable, end-to-end ML workflow for pan-European wildfire-probability mapping. Harnessing open data (Copernicus FWI, MODIS burn-area, HARCI-EU infrastructure, OSM), Dask and IBM CLAIMED, the pipeline automates:

- **Data ingestion & harmonization** into a distributed Zarr cube  
- **Baseline modeling** with logistic regression + Monte Carlo uncertainty  
- **Advanced training** of monotonic-constraint XGBoost, calibrated via Optuna  
- **Batch inference** for historical (2001–2022) and scenario (RCP4.5/8.5, 2023–2050) outputs at 2.5 km resolution  

Containerless operators and Airflow orchestration ensure traceable, CI/CD-friendly execution, while MinIO/S3 provides robust object storage.  
Built by **Alpha-Klima**.

---


## 🗂️ Project files

| Path                                           | Category       | Description                                                    |
|------------------------------------------------|----------------|----------------------------------------------------------------|
| **operators/**                                 | directory      | Python entrypoints—each compiled into a containerless CLAIMED operator. |
| ├─ `create_training_zarr.py`                   | operator       | Builds the multi-source Zarr training cube.                    |
| ├─ `train_logistic.py`                         | operator       | Fits the logistic-regression baseline model.                   |
| ├─ `logistic_prediction.py`                    | operator       | Runs batch inference + Monte Carlo uncertainty for the logistic baseline. |
| ├─ `optimize_xgb_hyperparameters_from_df.py`   | operator       | Bayesian hyperparameter tuning via Optuna TPE.                 |
| ├─ `training_xgboost.py`                       | operator       | Trains XGBoost with monotonic constraints & beta calibration.  |
| └─ `xgboost_prediction.py`                     | operator       | Executes tiled XGBoost inference (historical & scenario data). |
| **akfire_claimed_dag.py**                      | Airflow DAG    | Orchestrates the six CLAIMED operators end-to-end.             |
| **build_components.sh**                        | shell script   | Compiles & unzips each operator into `<name>:<version>/runneable.py`. |
| **config.json**                                | template       | Template for I/O paths, S3 credentials, temporal bounds & ML parameters. |
| **README.md**                                  | documentation  | Project overview, setup & usage instructions. 


---

## 📦 Installation

```bash
pip install claimed==0.1.9
pip install claimed-cli==0.1.6
pip install git+https://github.com/claimed-framework/c3.git
```

---

##  🧠 Why CLAIMED?

CLAIMED is a modular, reproducible, operator-based execution framework designed to work seamlessly with orchestration tools like Apache Airflow. Its main benefits are:

- **Modular design**: Reusable components (operators) with explicit dependencies
- **Version control**: Fully traceable builds with rollback support
- **Scalability**: Compatible with Dask and object stores like MinIO
- **Easy CI/CD**: Containerless operators simplify packaging and deployment

---

## 🛠️ Build Operators

To build an operator, we write a dedicated Python script placed in the **operators/** folder. Each script corresponds to a specific operator definition.

To build a claimed operator, run:

```bash
c3_create_containerless_operator -v 0.1.0 operators/create_training_zarr.py
```

In this example, **create_training_zarr.py** is the script defining the operator.

This command generates:

A **.cwl** metadata file in the same **operators/** folder

A ZIP archive in the **claimed-operators/** directory:
**claimed-operators/create_training_zarr:0.1.0.zip**

Repeat this process for each operator script you want to package.





## 📂 Why You Must Unzip Operator ZIPs

CLAIMED expects a runneable.py inside a directory path that matches the operator name. The CLI **does not look inside the ZIP**, so you must extract it:

```bash
unzip claimed-operators/create_training_zarr:0.1.0.zip -d claimed-operators/create_training_zarr:0.1.0
```

Now the CLAIMED CLI can locate runneable.py:

```bash
claimed --component containerless/create_training_zarr:0.1.0
```
---

## 🔄 Automate Operator Builds

Use a script like **build_components.sh**, this runs c3_create_containerless_operator -v *version* for every file in operators/ and unzips each ZIP into claimed-operators/*name*:*version*/.

---

## 🌍 Required Environment Variables

CLAIMED requires specific environment variables to locate configuration files, access datasets, and run operators correctly.

For our pipeline, which uses **MinIO** for object storage, set the following credentials and connection info:

```bash
export AKFIRE_ACCESS_KEY="..."
export AKFIRE_SECRET_KEY="..."
export AKFIRE_BUCKET="ak-fire-v1-0"
export AKFIRE_ENDPOINT=""
```

You must also set the path to the main configuration file used by the pipeline. This **config.json** includes settings for credentials, dataset handling, Zarr creation, training, and prediction:

```bash
export config="/path/to/config.json"
```

To run CLAIMED containerless operators, define the following paths:

```bash
export CLAIMED_COMPONENTS_DIR="$PWD"
export CLAIMED_CONTAINERLESS_OPERATOR_PATH="$PWD/claimed-operators/create_training_zarr:0.1.0"
```

CLAIMED_COMPONENTS_DIR tells CLAIMED where to find the claimed-operators/ directory. Typically, set it to the parent directory containing that folder ($PWD if you're already in the root).

CLAIMED_CONTAINERLESS_OPERATOR_PATH points to the specific operator package to run. This must be updated per operator.
For example, for the create_training_zarr operator:

```bash
export CLAIMED_CONTAINERLESS_OPERATOR_PATH="$PWD/claimed-operators/create_training_zarr:0.1.0"
```

This path should contain the runnable.py used by CLAIMED to execute the operator logic.

---

## 🚀 Integration with Apache Airflow

In our Airflow DAG (`akfire_claimed_dag.py`), each CLAIMED operator is executed using a `BashOperator`.

The DAG dynamically sets up all required environment variables and resolves paths, including MinIO credentials and the specific operator folder to run. It also downloads the required `config.json` from MinIO before running any task.

Each step of the pipeline (like `create_training_zarr`, `train_logistic`, etc.) is defined as a `BashOperator` running:

```bash
claimed --component containerless/<component_name>:<version>
```

Here’s an example for one task:

```python
BashOperator(
    task_id="create_training_zarr",
    bash_command="claimed --component containerless/create_training_zarr:0.1.0",
    env={
        **ENV_VARS,
        "CLAIMED_CONTAINERLESS_OPERATOR_PATH": op_path("create_training_zarr"),
        "config": str(config_file),
    },
)
```

## 🧰 Local Runner alternative

If you’re not using Airflow, here’s a standalone Python script to execute every operator in sequence:

```python
import os, shutil, subprocess
from pathlib import Path
from dataset_generation.utilities.s3_utilities import get_s3_fs

THIS_DIR = Path(__file__).resolve().parent
COMPONENTS_DIR = THIS_DIR / "claimed-operators"
VERSION = "0.1.0"

S3 = {
    "AKFIRE_ACCESS_KEY": os.getenv("AKFIRE_ACCESS_KEY"),
    "AKFIRE_SECRET_KEY": os.getenv("AKFIRE_SECRET_KEY"),
    "AKFIRE_BUCKET": os.getenv("AKFIRE_BUCKET"),
    "AKFIRE_ENDPOINT": os.getenv("AKFIRE_ENDPOINT"),
}

def download_config(tmp: Path) -> Path:
    fs = get_s3_fs()
    tmp.mkdir(exist_ok=True)
    cfg = tmp / "config.json"
    fs.get(f"{S3['AKFIRE_BUCKET']}/config.json", str(cfg))
    return cfg

def run_operator(name: str, cfg: Path):
    env = os.environ.copy()
    env.update(S3)
    env["config"] = str(cfg)
    env["CLAIMED_COMPONENTS_DIR"] = str(COMPONENTS_DIR)
    cmd = ["claimed", "--component", f"containerless/{name}:{VERSION}"]
    subprocess.run(cmd, env=env, check=True)

def main():
    tmp = THIS_DIR / "tmp"
    cfg = download_config(tmp)
    for op in [
        "create_training_zarr",
        "train_logistic",
        "logistic_prediction",
        "optimize_xgb_hyperparameters_from_df",
        "training_xgboost",
        "xgboost_prediction",
    ]:
        print(f"Running {op}…")
        run_operator(op, cfg)
    shutil.rmtree(tmp)

if __name__ == "__main__":
    main()

```