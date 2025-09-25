import itertools
from pathlib import Path
from benchmark.main import main
import pytest
import sys

CONFIG_FILES = [
    # "configs/tests/benchmark_v2_simple.yaml",
    "configs/tests/dofa_large_patch16_224_upernetdecoder_true_modified.yaml",
    "configs/tests/terratorch-iterate-configs/test_case_02/test_config_util__encoderdecoder_eo_v2_300_model_factory.yaml",
    "configs/tests/terratorch-iterate-configs/test_case_03/test_config_util__encoder_decoder_timm_resnet101_model_factory.yaml",
]
HPO = [True]
INPUT_TEST_MAIN = list(
    itertools.product(HPO, CONFIG_FILES)
)


def get_test_ids() -> list[str]:
    test_case_ids = list()
    for hpo, config in INPUT_TEST_MAIN:
        # get the filename
        filename = config.split("/")[-1].replace(".yaml", "")
        # set test id
        tid = f"{filename}_hpo_{hpo}"
        # append to list of test ids
        test_case_ids.append(tid)
    return test_case_ids


@pytest.mark.parametrize(
    "hpo, config",
    INPUT_TEST_MAIN,
    ids=get_test_ids(),
)
def test_main(
    hpo: bool,
    config: str,
):
    home_dir = Path(__file__).parent.parent.parent
    config_file: Path = home_dir / config
    assert config_file.exists()
    arguments = ["terratorch", "--config", str(config_file.resolve())]
    if hpo:
        arguments.insert(1, "--hpo")
    sys.argv = arguments
    main()
