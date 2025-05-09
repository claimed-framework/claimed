import subprocess
from pathlib import Path
from typing import Optional
from tests.test_benchmark import TEST_CASE_IDS
import click

@click.command()
@click.option('--test_id', default=None, help='test ID')
def run_tests(test_id: Optional[str] = None):
    if test_id is None:
        test_ids = TEST_CASE_IDS
    else:
        test_ids = [test_id]
    for tc_id in test_ids:
        print(f"Running test case {tc_id}")
        stderr_file = f"test-benchmark-{tc_id}.err"
        stdout_file = f"test-benchmark-{tc_id}.out"

        err_file = Path.home() / stderr_file
        # delete file if it exists
        if err_file.exists():
            print(f"Delete file {err_file}")
            err_file.unlink(missing_ok=True)
            assert not err_file.exists()
        out_file = Path.home() / stdout_file

        # delete file if it exists
        if out_file.exists():
            print(f"Delete file {out_file}")
            out_file.unlink(missing_ok=True)
            assert not out_file.exists()
        jbsub = f"jbsub -e {err_file} -o {out_file} -m 40G -c 1+1 -r v100 pytest -vv --cov-report html --cov=benchmark tests/test_benchmark.py::test_run_benchmark[{tc_id}]"
        cmd = jbsub.split()
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            print(f"Command executed successfully: {jbsub}")

        else:
            print(f"Command failed: {jbsub}")
            print("Command failed with error code:", result.returncode)
            print("stderr:", result.stderr)


if __name__ == "__main__":
    run_tests()
