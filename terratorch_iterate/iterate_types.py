"""
This module defines all the types expected at input. Used for type checking by jsonargparse.
"""

from ast import Dict
import copy
import enum
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Union, TYPE_CHECKING
from terratorch.tasks import (
    ClassificationTask,
    MultiLabelClassificationTask,
    PixelwiseRegressionTask,
    SemanticSegmentationTask,
    ObjectDetectionTask,
)
from torchgeo.datamodules import BaseDataModule

import logging

try:
    from geobench_v2.datamodules import GeoBenchDataModule
    GEOBENCH_AVAILABLE = True
except ImportError:
    GeoBenchDataModule = None  # type: ignore
    GEOBENCH_AVAILABLE = False
    logging.getLogger("terratorch").debug("geobench_v2 not installed")
    

if TYPE_CHECKING:
    from geobench_v2.datamodules import GeoBenchDataModule

valid_task_types = type[
    SemanticSegmentationTask
    | ClassificationTask
    | PixelwiseRegressionTask
    | ObjectDetectionTask
]


@dataclass
class TaskTypeEnum(enum.Enum):
    """
    Enum for the type of task to be performed. segmentation, regression or classification.
    """

    segmentation = "segmentation"
    regression = "regression"
    classification = "classification"
    multilabel_classification = "multilabel_classification"
    object_detection = "object_detection"

    def get_class_from_enum(
        self,
    ) -> valid_task_types:
        match self.value:
            case TaskTypeEnum.segmentation.value:
                return SemanticSegmentationTask
            case TaskTypeEnum.regression.value:
                return PixelwiseRegressionTask
            case TaskTypeEnum.classification.value:
                return ClassificationTask
            case TaskTypeEnum.multilabel_classification.value:
                return MultiLabelClassificationTask
            case TaskTypeEnum.object_detection.value:
                return ObjectDetectionTask
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


optimization_space_type = dict[
    str, Union[list, ParameterBounds, "optimization_space_type"]
]


@dataclass
class Defaults:
    """
    Default parameters set for each of the tasks.

    These parameters will be combined with task specific ones to form the final parameters for the Terratorch training.

    Args:
        trainer_args (dict): Arguments passed to Lightning Trainer.
        terratorch_task (dict): Arguments for the Terratorch Task.
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
        terratorch_task (dict): Arguments for the Terratorch Task.
        datamodule (BaseDataModule  | GeoBenchDataModule): Datamodule to be used.
        direction (str): One of min or max. Direction to optimize the metric in.
        metric (str): Metric to be optimized. Defaults to "val/loss".
        early_prune (bool): Whether to prune unpromising runs early. Defaults to False.
        early_stop_patience (int, None): Whether to use Lightning early stopping of runs. Defaults to None, which does not do early stopping.
        optimization_except (str[str]): HyperParameters from the optimization space to be ignored for this task.
        max_run_duration (str, None): maximum allowed run duration in the form DD:HH:MM:SS; will stop a run after this
            amount of time. Defaults to None, which doesn't stop runs by time.
    """

    name: str
    type: TaskTypeEnum = field(repr=False)
    datamodule: Union[BaseDataModule, "GeoBenchDataModule"] = field(repr=False)
    direction: str
    terratorch_task: Optional[dict[str, Any]] = None
    metric: str = "val/loss"
    early_prune: bool = False
    early_stop_patience: int | None = None
    optimization_except: set[str] = field(default_factory=set)
    max_run_duration: str | None = None


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
    """
    Combine task-specific parameters with default parameters.

    Args:
        task (Task): Task object containing task-specific parameters.
        defaults (Defaults): Defaults object containing default parameters.

    Returns:
        TrainingSpec: TrainingSpec object containing combined parameters.
    """
    terratorch_task: Optional[Dict[str, Any]] = copy.deepcopy(defaults.terratorch_task)
    if terratorch_task is None:
        terratorch_task = {}
    if task.terratorch_task is None:
        task.terratorch_task = {}
    # merge task specific args with default args
    recursive_merge(terratorch_task, task.terratorch_task)
    task_with_defaults = replace(task, terratorch_task=terratorch_task)
    return TrainingSpec(task_with_defaults, defaults.trainer_args)
