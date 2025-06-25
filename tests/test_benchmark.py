import itertools
import logging
from benchmark.benchmark_types import Defaults, Task, TaskTypeEnum
import pytest
from benchmark.backbone_benchmark import benchmark_backbone
from terratorch.datamodules import MChesapeakeLandcoverNonGeoDataModule
from albumentations import HorizontalFlip, VerticalFlip, Resize
from albumentations.pytorch.transforms import ToTensorV2
import os
from pathlib import Path
import uuid
from jsonargparse import ArgumentParser


BACKBONE_PRETRAINED_FILE = os.getenv(
    "BACKBONE_PRETRAINED_FILE",
    "/dccstor/geofm-finetuning/pretrain_ckpts/v9_no_sea/vit_b/epoch-395-loss-0.0339_clean.pt",
)

SEGMENTATION_V1 = os.getenv(
    "SEGMENTATION_V1", "/dccstor/geofm-finetuning/datasets/geobench/segmentation_v1.0"
)

# OUTPUT_DIR = os.getenv(
#     "OUTPUT_DIR", "/dccstor/geofm-finetuning/terratorch-iterate-test/"
# )

RAY_STORAGE = os.getenv(
    "RAY_STORAGE", "/dccstor/geofm-finetuning/terratorch-iterate-test/ray_storage"
)


@pytest.fixture(scope="module")
def defaults() -> Defaults:
    file = BACKBONE_PRETRAINED_FILE
    assert Path(file).exists(), f"Error! {file=} does not exist"
    trainer_args = {
        "precision": "bf16-mixed",
        "max_epochs": 10,
    }
    terratorch_task = {
        "model_args": {
            "pretrained": True,
            "backbone": "prithvi_vit_100",
            "backbone_out_indices": [2, 5, 8, 11],
            "backbone_pretrained_cfg_overlay": {"file": file},
        },
        "model_factory": "PrithviModelFactory",
        "optimizer": "AdamW",
    }
    return Defaults(trainer_args=trainer_args, terratorch_task=terratorch_task)


@pytest.fixture(scope="module")
def mchesapeakelandcovernongeodatamodule() -> MChesapeakeLandcoverNonGeoDataModule:
    data_root = SEGMENTATION_V1
    assert Path(data_root).exists(), f"Error! Directory {data_root} does not exist"
    train_transform = [Resize(height=224, width=224), ToTensorV2()]
    test_transform = [
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Resize(height=224, width=224),
        ToTensorV2(),
    ]
    return MChesapeakeLandcoverNonGeoDataModule(
        num_workers=6,
        batch_size=16,
        partition="0.10x_train",
        train_transform=train_transform,
        test_transform=test_transform,
        data_root=data_root,
        bands=["RED", "GREEN", "BLUE", "NIR"],
    )


@pytest.fixture(scope="module")
def tasks(mchesapeakelandcovernongeodatamodule):

    t = Task(
        name="chesapeake",
        type=TaskTypeEnum.segmentation,
        direction="max",
        metric="val/Multiclass_Jaccard_Index",
        early_stop_patience=10,
        terratorch_task={
            "loss": "ce",
            "model_args": {
                "decoder": "UperNetDecoder",
                "decoder_channels": 128,
                "decoder_scale_modules": True,
                "bands": ["RED", "GREEN", "BLUE", "NIR"],
                "num_classes": 7,
            },
        },
        datamodule=mchesapeakelandcovernongeodatamodule,
    )
    return [t]


def find_file(directory: str, filename: str):
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


CONFIG_FILES = [
    "configs/tests/geobench_v1_resnet_cashew.yaml",
    "configs/tests/geobench_v1_prithvi_cashew.yaml",
    "configs/tests/benchmark_v2_simple.yaml",
    "configs/tests/dofa_large_patch16_224_upernetdecoder_true_modified.yaml",
    "configs/tests/geobench_v1_ssl4eos12_resnet50_sentinel2_all_moco_smp_unet_true.yaml",
    "configs/nasabench_vit_b_os.yaml",
]
CONTINUE_EXISTING_EXPERIMENT = [True, False]
TEST_MODELS = [True, False]
INPUT_TEST_RUN_BENCHMARK = list(
    itertools.product(CONFIG_FILES, CONTINUE_EXISTING_EXPERIMENT, TEST_MODELS)
)
TEST_CASE_IDS = [str(i) for i in range(0, len(INPUT_TEST_RUN_BENCHMARK))]


@pytest.mark.parametrize(
    "config, continue_existing_experiment, test_models",
    INPUT_TEST_RUN_BENCHMARK,
    ids=TEST_CASE_IDS,
)
def test_run_benchmark(
    config: str, continue_existing_experiment: bool, test_models: bool
):
    path = os.path.join(os.getcwd(), config)
    config_path = Path(path)
    # instantiate objects from yaml
    parser = ArgumentParser()
    parser.add_argument('--defaults', type=Defaults)  # to ignore model
    parser.add_argument('--optimization_space', type=dict)  # to ignore model
    parser.add_argument('--experiment_name', type=str)  # to ignore model
    parser.add_argument('--run_name', type=str)  # to ignore model
    parser.add_argument('--save_models', type=bool)  # to ignore model
    parser.add_argument('--storage_uri', type=str)  # to ignore model
    parser.add_argument('--ray_storage_path', type=str)  # to ignore model
    parser.add_argument('--n_trials', type=int)  # to ignore model
    parser.add_argument('--run_repetitions', type=int)  # to ignore model
    parser.add_argument('--tasks', type=list[Task])
    config = parser.parse_path(str(config_path))
    config_init = parser.instantiate_classes(config)
    # validate the objects
    experiment_name = config_init.experiment_name
    experiment_name = f"{experiment_name}_continue_{continue_existing_experiment}_test_models_{test_models}"
    assert isinstance(experiment_name, str), f"Error! {experiment_name=} is not a str"
    run_name = config_init.run_name
    if run_name is not None:
        assert isinstance(run_name, str), f"Error! {run_name=} is not a str"
    tasks = config_init.tasks
    assert isinstance(tasks, list), f"Error! {tasks=} is not a list"
    for t in tasks:
        assert isinstance(t, Task), f"Error! {t=} is not a Task"
    defaults = config_init.defaults
    assert isinstance(defaults, Defaults), f"Error! {defaults=} is not a Defaults"
    # defaults.trainer_args["max_epochs"] = 5
    storage_uri = config_init.storage_uri
    assert isinstance(storage_uri, str), f"Error! {storage_uri=} is not a str"
    storage_uri_path = Path(storage_uri) / uuid.uuid4().hex / "hpo"
    if not storage_uri_path.exists():
        try:
            storage_uri_path.mkdir(parents=True, exist_ok=True)
            print(f"Directory created at: {path}")
        except FileNotFoundError as e:
            print(f"Error creating directory: {e}")

    optimization_space = config_init.optimization_space
    assert isinstance(
        optimization_space, dict
    ), f"Error! {optimization_space=} is not a dict"
    ray_storage = RAY_STORAGE
    assert isinstance(ray_storage, str), f"Error! {ray_storage=} is not a str"
    ray_storage_path = Path(ray_storage) / uuid.uuid4().hex
    if not ray_storage_path.exists():
        try:
            ray_storage_path.mkdir(parents=True, exist_ok=True)
            print(f"Directory created at: {path}")
        except FileNotFoundError as e:
            print(f"Error creating directory: {e}")
    n_trials = config_init.n_trials
    assert isinstance(n_trials, int) and n_trials > 0, f"Error! {n_trials=} is invalid"
    # run_repetions is an optional parameter
    run_repetitions = config_init.run_repetitions
    if run_repetitions is not None:
        assert (
            isinstance(run_repetitions, int) and run_repetitions >= 0
        ), f"Error! {run_repetitions=} is invalid"
    else:
        run_repetitions = 0
    mlflow_info = benchmark_backbone(
        experiment_name=experiment_name,
        run_name=run_name,
        run_id=None,
        defaults=defaults,
        tasks=tasks,
        n_trials=n_trials,
        save_models=False,
        storage_uri=str(storage_uri_path),
        ray_storage_path=str(ray_storage_path),
        optimization_space=optimization_space,
        continue_existing_experiment=continue_existing_experiment,
        test_models=test_models,
        run_repetitions=run_repetitions,
        logger=None,
    )
    assert isinstance(mlflow_info, dict), f"Error! {mlflow_info=} is not a dict"
    validate_results(
        experiment_name=experiment_name,
        storage_uri=str(storage_uri_path),
        finished_run_id=mlflow_info["experiment_id"],
    )


@pytest.mark.parametrize(
    "config, continue_existing_experiment, test_models",
    [
        ("configs/tests/benchmark_marida_l2a_terramind_base.yaml", False, False),
    ],
)
def test_run_benchmark_no_specific_terratorch_task(
    config: str, continue_existing_experiment: bool, test_models: bool
):

    path = os.path.join(os.getcwd(), config)
    config_path = Path(path)
    assert (
        config_path.exists()
    ), f"Error! config does not exist: {config_path.resolve()}"
    # instantiate objects from yaml
    parser = ArgumentParser()
    parser.add_argument('--defaults', type=Defaults)  # to ignore model
    parser.add_argument('--optimization_space', type=dict)  # to ignore model
    parser.add_argument('--experiment_name', type=str)  # to ignore model
    parser.add_argument('--run_name', type=str)  # to ignore model
    parser.add_argument('--save_models', type=bool)  # to ignore model
    parser.add_argument('--storage_uri', type=str)  # to ignore model
    parser.add_argument('--ray_storage_path', type=str)  # to ignore model
    parser.add_argument('--n_trials', type=int)  # to ignore model
    parser.add_argument('--run_repetitions', type=int)  # to ignore model
    parser.add_argument('--tasks', type=list[Task])
    config = parser.parse_path(str(config_path))
    config_init = parser.instantiate_classes(config)
    # validate the objects
    experiment_name = config_init.experiment_name
    experiment_name = f"{experiment_name}_continue_{continue_existing_experiment}_test_models_{test_models}"
    assert isinstance(experiment_name, str), f"Error! {experiment_name=} is not a str"
    run_name = config_init.run_name
    if run_name is not None:
        assert isinstance(run_name, str), f"Error! {run_name=} is not a str"
    tasks = config_init.tasks
    assert isinstance(tasks, list), f"Error! {tasks=} is not a list"
    for t in tasks:
        assert isinstance(t, Task), f"Error! {t=} is not a Task"
        if t.terratorch_task is not None:
            t.terratorch_task = None

    defaults = config_init.defaults
    assert isinstance(defaults, Defaults), f"Error! {defaults=} is not a Defaults"
    # defaults.trainer_args["max_epochs"] = 5
    storage_uri = config_init.storage_uri
    assert isinstance(storage_uri, str), f"Error! {storage_uri=} is not a str"
    storage_uri_path = Path(storage_uri) / uuid.uuid4().hex / "hpo"
    if not storage_uri_path.exists():
        try:
            storage_uri_path.mkdir(parents=True, exist_ok=True)
            print(f"Directory created at: {path}")
        except FileNotFoundError as e:
            print(f"Error creating directory: {e}")
    optimization_space = config_init.optimization_space
    assert isinstance(
        optimization_space, dict
    ), f"Error! {optimization_space=} is not a dict"
    ray_storage = RAY_STORAGE
    assert isinstance(ray_storage, str), f"Error! {ray_storage=} is not a str"
    ray_storage_path = Path(ray_storage) / uuid.uuid4().hex
    if not ray_storage_path.exists():
        try:
            ray_storage_path.mkdir(parents=True, exist_ok=True)
            print(f"Directory created at: {path}")
        except FileNotFoundError as e:
            print(f"Error creating directory: {e}")
    n_trials = config_init.n_trials
    assert isinstance(n_trials, int) and n_trials > 0, f"Error! {n_trials=} is invalid"
    # run_repetions is an optional parameter
    run_repetitions = config_init.run_repetitions
    if run_repetitions is not None:
        assert (
            isinstance(run_repetitions, int) and run_repetitions >= 0
        ), f"Error! {run_repetitions=} is invalid"
    else:
        run_repetitions = 0
    finished_run_id = benchmark_backbone(
        experiment_name=experiment_name,
        run_name=run_name,
        run_id=None,
        defaults=defaults,
        tasks=tasks,
        n_trials=n_trials,
        save_models=False,
        storage_uri=str(storage_uri_path),
        ray_storage_path=str(ray_storage_path),
        optimization_space=optimization_space,
        continue_existing_experiment=continue_existing_experiment,
        test_models=test_models,
        run_repetitions=run_repetitions,
    )
    validate_results(
        experiment_name=experiment_name,
        storage_uri=str(storage_uri_path),
        finished_run_id=finished_run_id,
    )


def validate_results(experiment_name: str, storage_uri: str, finished_run_id: str):
    # get the most recent modified directory
    dir_path = Path(storage_uri) / finished_run_id
    assert dir_path.exists(), f"Error! Directory does not exist: {dir_path}"
    # find mlflow.runName files within the result dir
    meta_yaml = "meta.yaml"

    meta_yaml_path = dir_path / meta_yaml
    assert (
        meta_yaml_path.exists()
    ), f"Error! meta.yaml file {meta_yaml_path} does not exist"
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
        assert (
            experiment_name_found and experiment_id_found
        ), f"Error! Both experiment name ({experiment_name=}) and finished run id ({finished_run_id=}) must be in the {meta_yaml_path=}: {experiment_id_found=} {experiment_name_found=}"
    # TODO delete the directories that were created by this test case
