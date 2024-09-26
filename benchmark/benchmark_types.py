"""
This module defines all the types expected at input. Used for type checking by jsonargparse.
"""
import copy
import enum
from dataclasses import dataclass, field, replace
from typing import Any, Union

import torch
from lightning import Trainer
from terratorch.datasets import HLSBands
from terratorch.tasks import (
    ClassificationTask,
    MultiLabelClassificationTask,
    PixelwiseRegressionTask,
    SemanticSegmentationTask,
)
from torchgeo.datamodules import BaseDataModule

valid_task_types = type[
    SemanticSegmentationTask | ClassificationTask | PixelwiseRegressionTask
]


class TaskTypeEnum(enum.Enum):
    """
    Enum for the type of task to be performed. segmentation, regression or classification.
    """

    segmentation = "segmentation"
    regression = "regression"
    classification = "classification"
    multilabel_classification = "multilabel_classification"

    def get_class_from_enum(
        self,
    ) -> valid_task_types:
        match self:
            case TaskTypeEnum.segmentation:
                return SemanticSegmentationTask
            case TaskTypeEnum.regression:
                return PixelwiseRegressionTask
            case TaskTypeEnum.classification:
                return ClassificationTask
            case TaskTypeEnum.multilabel_classification:
                return MultiLabelClassificationTask
            case _:
                raise TypeError("Task type does not exist")


class ParameterTypeEnum(enum.Enum):
    """
    Enum for the type of parameter allowed in ParameterBounds. integer or real.
    """

    integer = "int"
    real = "real"


@dataclass
class ParameterBounds:
    """
    Dataclass defining a numerical range to search over.

    Args:
        min (float | int): Minimum.
        max (float | int): Maximum.
        type (ParameterTypeEnum): Whether the range is in the space of integers or real numbers.
        log (bool): Whether to search over the log space (useful for parameters that vary wildly in scale, e.g. learning rate)
    """

    min: float | int
    max: float | int
    type: ParameterTypeEnum
    log: bool = False

    def __post_init__(self):
        if not isinstance(self.type, ParameterTypeEnum):
            self.type = ParameterTypeEnum(self.type)

optimization_space_type = dict[str, Union[list, ParameterBounds, 'optimization_space_type']]


@dataclass
class Defaults:
    """
    Default parameters set for each of the tasks. 

    These parameters will be combined with task specific ones to form the final parameters for the Terratorch training.

    Args:
        trainer_args (dict): Arguments passed to Lightning Trainer.
        task (dict): Arguments for the Terratorch Task.
    """

    trainer_args: dict[str, Any] = field(default_factory=dict)
    terratorch_task: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """
    Parameters passed to define each of the tasks.

    These parameters are combined with any specified defaults to generate the final task parameters.

    Args:
        name (str): Name for this task
        type (TaskTypeEnum): Type of task.
        task (dict): Arguments for the Terratorch Task.
        datamodule (BaseDataModule): Datamodule to be used.
    """

    name: str
    type: TaskTypeEnum
    terratorch_task: dict[str, Any]
    datamodule: BaseDataModule
    direction: str
    metric: str = "val/loss"
    early_prune: bool = False
    early_stop_patience: int | None = None
    optimization_except: set[str] = field(default_factory=set)

@dataclass
class TrainingSpec:
    task: Task
    trainer_args: dict[str, Any] = field(default_factory=dict)

def recursive_merge(first_dict: dict[str, Any], second_dict: dict[str, Any]):
    # consider using deepmerge instead of this
    for key, val in second_dict.items():
        if key not in first_dict:
            first_dict[key] = val
        else:
            # if it is a dictionary, recurse deeper
            if isinstance(val, dict):
                recursive_merge(first_dict[key], val)
            # if it is not further nested, just replace the value
            else:
                first_dict[key] = val

def combine_with_defaults(task: Task, defaults: Defaults) -> TrainingSpec:
    terratorch_task = copy.deepcopy(defaults.terratorch_task)
    recursive_merge(terratorch_task, task.terratorch_task)
    task_with_defaults = replace(task, terratorch_task=terratorch_task)
    return TrainingSpec(task_with_defaults, defaults.trainer_args)