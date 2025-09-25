import subprocess
from pathlib import Path
from typing import Optional
from tests.test_benchmark import get_test_ids
import click

# rm geobench_v1_prithvi* && bsub -e ~/geobench_v1_prithvi.err -o ~/geobench_v1_prithvi.out -M 40G -gpu "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" terratorch iterate --hpo --config configs/geobench_v1_prithvi.yaml

REPO_HOME_DIR = Path(__file__).parent
LOGS_DIR = REPO_HOME_DIR / "logs"

if not LOGS_DIR.exists():
    LOGS_DIR.mkdir()

@click.group()
def cli():
    pass


def submit_job(
    stderr_file: str,
    stdout_file: str,
    tc_id: str | None = None,
    config: str | None = None,
):
    err_file = LOGS_DIR / stderr_file
    # delete file if it exists
    if err_file.exists():
        print(f"Delete file {err_file}")
        err_file.unlink(missing_ok=True)
        assert not err_file.exists()

    out_file = LOGS_DIR / stdout_file
    # delete file if it exists
    if out_file.exists():
        print(f"Delete file {out_file}")
        out_file.unlink(missing_ok=True)
        assert not out_file.exists()
    if tc_id is not None:
        jbsub = f"bsub -e {err_file} -o {out_file} -M 40G -gpu \"num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB\" pytest -vv tests/test_benchmark.py::test_run_benchmark[{tc_id}]"
    elif config is not None:
        jbsub = f"bsub -e {err_file} -o {out_file} -M 40G -gpu \"num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB\" terratorch iterate --hpo --config {config}"
    else:
        raise ValueError("Error! Either tc_id or config must be not None")
    cmd = jbsub.split()
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0:
        print(f"Command executed successfully: {jbsub}")

    else:
        print(f"Command failed: {jbsub}")
        print("Command failed with error code:", result.returncode)
        print("stderr:", result.stderr)


@click.command()
@click.option('--test_id', default=None, help='test ID')
def run_tests(test_id: Optional[str] = None):
    if test_id is None:
        test_ids = get_test_ids()
    else:
        test_ids = [test_id]
    for tc_id in test_ids:
        print(f"Running test case: tests/test_benchmark.py::test_run_benchmark {tc_id}")
        stderr_file = f"{tc_id}.err"
        stdout_file = f"{tc_id}.out"

        submit_job(stderr_file=stderr_file, stdout_file=stdout_file, tc_id=tc_id)


@click.command()
@click.option('--config', default=None, help='path to config file')
def run_job(config: str):
    home_dir = Path(__file__).parent
    config_path = home_dir / config
    assert config_path.exists()
    stem = config_path.stem
    err_file = f"{stem}.err"
    out_file = f"{stem}.out"
    submit_job(stdout_file=out_file, stderr_file=err_file, config=config)


cli.add_command(run_job)
cli.add_command(run_tests)

if __name__ == '__main__':
    cli()
