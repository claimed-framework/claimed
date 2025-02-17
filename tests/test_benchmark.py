from typing import List
from benchmark.benchmark_types import Defaults, Task, TaskTypeEnum
import pytest
from benchmark.backbone_benchmark import benchmark_backbone
from terratorch.datamodules import MChesapeakeLandcoverNonGeoDataModule
from albumentations import HorizontalFlip, VerticalFlip, Resize
from albumentations.pytorch.transforms import ToTensorV2
import uuid
import os
from pathlib import Path


BACKBONE_PRETRAINED_FILE = os.getenv(
    "BACKBONE_PRETRAINED_FILE",
    "/dccstor/geofm-finetuning/pretrain_ckpts/v9_no_sea/vit_b/epoch-395-loss-0.0339_clean.pt",
)

SEGMENTATION_V1 = os.getenv(
    "SEGMENTATION_V1", "/dccstor/geofm-finetuning/datasets/geobench/segmentation_v1.0"
)

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR", "/dccstor/geofm-finetuning/terratorch-iterate-test"
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


def test_run_benchmark(defaults: Defaults, tasks: List[Task]):
    storage_uri = OUTPUT_DIR
    assert Path(storage_uri), f"Error! directory {storage_uri} does not exist"
    ray_storage_path = None
    optimization_space = {
        "batch_size": [8, 32, 64],
        "lr": {"max": 1e-3, "min": 1e-6, "type": "real", "log": True},
        "optimizer_hparams": {"weight_decay": {"min": 0, "max": 0.4, "type": "real"}},
        "model_args": {"decoder_channels": [64, 128, 256]},
    }
    run_id = uuid.uuid4().hex
    experiment_name = f"test_chesapeake_segmentation_{run_id}"
    run_name = f"run_name_geobench_{run_id}"

    mlflow_experiment_id = benchmark_backbone(
        experiment_name=experiment_name,
        run_name=run_name,
        run_id=None,
        defaults=defaults,
        tasks=tasks,
        n_trials=2,
        save_models=False,
        storage_uri=storage_uri,
        ray_storage_path=ray_storage_path,
        optimization_space=optimization_space,
    )
    # get the most recent modified directory
    dir_path = Path(storage_uri) / mlflow_experiment_id
    assert dir_path.exists(), f"Error! directory does not exist: {dir_path}"
    # find mlflow.runName files within the result dir
    meta_yaml = "meta.yaml"

    meta_yaml_path = dir_path / meta_yaml
    assert meta_yaml_path.exists(), f"Error! {meta_yaml_path=} does not exist"
    # open file and check that the experiment name is the same
    with open(meta_yaml_path, mode="r") as f:
        lines = f.readlines()
        experiment_name_found = False
        experiment_id_found = False
        for line in lines:
            if experiment_name in line:
                experiment_name_found = True
            if experiment_id_found in line:
                experiment_id_found = True
        assert (
            experiment_name_found and experiment_id_found
        ), f"Error! {experiment_id_found=} {experiment_name_found=}"
    # TODO delete the directories that were created by this test case
