import subprocess
from pathlib import Path

STD_ERR_FILE="test-terratorch-iterate.err"
STD_OUT_FILE="test-terratorch-iterate.out"

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

    jbsub = f"jbsub -e {err_file} -o {out_file} -m 40G -c 1+1 -r v100 pytest --cov-report html --cov=benchmark tests/test_benchmark.py"
    print(f"Submitting: {jbsub}")
    subprocess.run(jbsub.split())


if __name__ == "__main__":
    main()