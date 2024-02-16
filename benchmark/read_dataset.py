import enum
import sys
from dataclasses import dataclass, field
from typing import Any

import albumentations
import terratorch
import torch
from jsonargparse import CLI
from lightning import LightningDataModule, Trainer
from terratorch.datasets import HLSBands
from terratorch.models import PrithviModelFactory
from terratorch.models.model import ModelFactory
from terratorch.tasks import (
    IBMClassificationTask,
    IBMPixelwiseRegressionTask,
    IBMSemanticSegmentationTask,
)
from torchgeo.trainers import BaseTask


class TaskTypeEnum(enum.Enum):
    segmentation = "segmentation"
    regression = "regression"
    classification = "classification"


def get_class_from_enum(task_type: TaskTypeEnum) -> BaseTask:
    match task_type:
        case TaskTypeEnum.segmentation:
            return IBMSemanticSegmentationTask
        case TaskTypeEnum.regression:
            return IBMPixelwiseRegressionTask
        case TaskTypeEnum.classification:
            return IBMClassificationTask
        case _:
            raise TypeError("Task type does not exist")


@dataclass
class Backbone:
    backbone_name: str
    model_factory: ModelFactory = field(default_factory=PrithviModelFactory)
    backbone_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    type: TaskTypeEnum
    bands: list[HLSBands | int]
    datamodule: LightningDataModule
    decoder_name: str
    loss: str
    freeze_backbone: bool = False
    decoder_args: dict[str, Any] = field(default_factory=dict)
    head_args: dict[str, Any] = field(default_factory=dict)
    ignore_index: int = None

def build_model_args(backbone: Backbone, task: Task):
    args = {}
    args["task"] = task.type
    args["backbone"] = backbone.backbone_name
    for backbone_key, backbone_val in backbone.backbone_args.items():
        args[f"backbone_{backbone_key}"] = backbone_val
    
    args["decoder"] = task.decoder_name
    for decoder_key, decoder_val in backbone.decoder_args.items():
        args[f"decoder_{decoder_key}"] = decoder_val
    
    for head_key, head_val in backbone.head_args.items():
        args[f"head_{head_key}"] = head_val
        
    args["in_channels"] = len(task.bands)
    args["bands"] = task.bands

def benchmark_backbone(backbone: Backbone, tasks: list[Task]):
    for task in tasks:
        trainer_class = get_class_from_enum(task.type)
        model_args = build_model_args(backbone, task)
        
        lr=1e-4
        trainer = trainer_class(model_args, backbone.model_factory, loss=task.loss, lr=lr, optimizer=torch.optim.AdamW, freeze_backbone=task.freeze_backbone, ignore_index=task.ignore_index)
        print(trainer)

CLI(benchmark_backbone)
