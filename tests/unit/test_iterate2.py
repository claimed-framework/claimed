import os

def test_iterate2( ):
    script = """
iterate \
  --script ./examples/bumpy_function.py \
  --root-dir . \
  --optuna-study-name hpo \
  --optuna-db-path "sqlite:///iterate_study.db" \
  --hpo-yaml examples/bumpy_hpo.yaml \
  --optuna-n-trials 10 \
  --wlm none \
  --metric yval \
  --gpu-count 0 \
  --cpu-count 1 \
  --mem-gb 1
    """
    ret = os.system(script)
    assert ret == 0