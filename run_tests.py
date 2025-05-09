import subprocess
from pathlib import Path
from tests.test_benchmark import TEST_CASE_IDS

STD_ERR_FILE = "test-terratorch-iterate.err"
STD_OUT_FILE = "test-terratorch-iterate.out"


def main():
    err_file = Path.home() / STD_ERR_FILE
    if err_file.exists():
        print(f"Delete file {err_file}")
        err_file.unlink(missing_ok=True)
        assert not err_file.exists()
    out_file = Path.home() / STD_OUT_FILE

    if out_file.exists():
        print(f"Delete file {out_file}")
        out_file.unlink(missing_ok=True)
        assert not out_file.exists()
    for tc_id in TEST_CASE_IDS:
        jbsub = f"jbsub -e {err_file} -o {out_file} -m 40G -c 1+1 -r v100 pytest --cov-report html --cov=benchmark tests/test_benchmark.py::test_run_benchmark[{tc_id}]"
        print(f"Submitting: {jbsub}")
        cmd = jbsub.split()
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            print("Command executed successfully:")
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
        else:
            print("Command failed with error code:", result.returncode)
            print("stderr:", result.stderr)


if __name__ == "__main__":
    main()
