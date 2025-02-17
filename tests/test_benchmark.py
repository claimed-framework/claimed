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

@pytest.fixture(scope="module")
def defaults() -> Defaults:
    file = "/dccstor/geofm-finetuning/pretrain_ckpts/v9_no_sea/vit_b/epoch-395-loss-0.0339_clean.pt"
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
            "backbone_pretrained_cfg_overlay": {
                "file": file
            },
        },
        "model_factory": "PrithviModelFactory",
        "optimizer": "AdamW",
    }
    return Defaults(trainer_args=trainer_args, terratorch_task=terratorch_task)


@pytest.fixture(scope="module")
def mchesapeakelandcovernongeodatamodule() -> MChesapeakeLandcoverNonGeoDataModule:
    data_root = "/dccstor/geofm-finetuning/datasets/geobench/segmentation_v1.0"
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


def get_most_recent_modified_dir(path):
    """
    Returns the most recently modified directory within the given path.
    """
    if not os.path.exists(path):
        raise ValueError(f"Path '{path}' does not exist.")

    if not os.path.isdir(path):
        raise ValueError(f"Path '{path}' is not a directory.")

    sub_dirs = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]
    if not sub_dirs:
        return None

    return max(sub_dirs, key=os.path.getmtime)


def find_file(directory: str, filename: str):
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def test_run_benchmark(defaults: Defaults, tasks: List[Task]):
    storage_uri = "/dccstor/geofm-finetuning/terratorch-iterate-test"
    assert Path(storage_uri), f"Error! directory {storage_uri} does not exist"
    ray_storage_path = "/dccstor/geofm-finetuning/carlosgomes/ray_storage"
    optimization_space = {
        "batch_size": [8, 32, 64],
        "lr": {"max": 1e-3, "min": 1e-6, "type": "real", "log": True},
        "optimizer_hparams": {"weight_decay": {"min": 0, "max": 0.4, "type": "real"}},
        "model_args": {"decoder_channels": [64, 128, 256]},
    }
    unique_id = uuid.uuid4().hex
    experiment_name = f"test_chesapeake_segmentation_{unique_id}"
    run_name = f"run_name_geobench_{unique_id}"
    benchmark_backbone(
        experiment_name=experiment_name,
        run_name=run_name,
        defaults=defaults,
        tasks=tasks,
        n_trials=2,
        save_models=False,
        storage_uri=storage_uri,
        ray_storage_path=ray_storage_path,
        optimization_space=optimization_space,
    )
    # get the most recent modified directory
    dir_path = get_most_recent_modified_dir(path=storage_uri)
    # find mlflow.runName files within the result dir
    mlflow_run_name = "mlflow.runName"
    mlflow_path = find_file(directory=dir_path, filename=mlflow_run_name)
    # open file and check that the experiment name is the same
    with open(mlflow_path, mode="r") as f:
        line = f.read()
        assert (
            run_name in line
        ), f"Error! {run_name=} is not part of {line=} from file={mlflow_path}"
    # TODO delete the directories that were created by this test case
