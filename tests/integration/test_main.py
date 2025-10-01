import itertools
from pathlib import Path

import yaml
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
INPUT_TEST_MAIN = list(itertools.product(HPO, CONFIG_FILES))


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


def validate_results(experiment_name: str, storage_uri: str, finished_run_id: str):
    # get the most recent modified directory
    dir_path = Path(storage_uri) / finished_run_id
    assert dir_path.exists(), f"Error! Directory does not exist: {dir_path}"
    # find mlflow.runName files within the result dir
    meta_yaml = "meta.yaml"

    meta_yaml_path = dir_path / meta_yaml
    assert meta_yaml_path.exists(), (
        f"Error! meta.yaml file {meta_yaml_path} does not exist"
    )
    # open file and check that the experiment name is the same
    with open(meta_yaml_path, mode="r") as f:
        # read all the lines
        lines = f.readlines()
        # try to find experiment id and name in these lines
        experiment_name_found: bool = False
        experiment_id_found: bool = False
        for line in lines:
            if experiment_name in line:
                experiment_name_found = True
            if finished_run_id in line:
                experiment_id_found = True
        assert experiment_name_found and experiment_id_found, (
            f"Error! Both experiment name ({experiment_name=}) and finished run id ({finished_run_id=}) must be in the {meta_yaml_path=}: {experiment_id_found=} {experiment_name_found=}"
        )
    # TODO delete the directories that were created by this test case


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
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)
    storage_uri: str = config_data["storage_uri"]
    # handling relative paths
    if storage_uri.startswith(".") or storage_uri.startswith(".."):
        repo_home_dir = Path(__file__).parent.parent.parent
        abs_path = repo_home_dir / storage_uri
        storage_uri = str(abs_path.resolve())
    experiment_name = config_data["experiment_name"]
    arguments = ["terratorch", "--config", str(config_file.resolve())]
    if hpo:
        arguments.insert(1, "--hpo")
    sys.argv = arguments
    # main only returns a dict when hpo is True
    mlflow_info = main()
    assert isinstance(mlflow_info, dict), f"Error! {mlflow_info=} is not a dict"
    validate_results(
        experiment_name=experiment_name,
        storage_uri=storage_uri,
        finished_run_id=mlflow_info["experiment_id"],
    )
