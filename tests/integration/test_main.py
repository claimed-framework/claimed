from pathlib import Path
from benchmark.main import main
import pytest
import sys


# terratorch iterate --hpo --config configs/tests/benchmark_v2_simple.yaml
@pytest.mark.parametrize(
    "hpo, config",
    [
        (
            True,
            "configs/tests/terratorch-iterate-configs/test_case_02/test_config_util__encoderdecoder_eo_v2_300_model_factory.yaml"
        )
    ],
)
def test_main(hpo: bool, config: str):
    home_dir = Path(__file__).parent.parent.parent 
    config_file: Path = home_dir / config
    assert config_file.exists()
    arguments = ["terratorch", "--config", str(config_file.resolve())]
    if hpo:
        arguments.insert(1, "--hpo")
    sys.argv = arguments
    main()
