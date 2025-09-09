from benchmark.benchmark_types import TaskTypeEnum
import pytest
from terratorch.tasks.base_task import TerraTorchTask
from terratorch.tasks.classification_tasks import ClassificationTask
from terratorch.tasks.multilabel_classification_tasks import (
    MultiLabelClassificationTask,
)
from terratorch.tasks.segmentation_tasks import SemanticSegmentationTask
from terratorch.tasks.regression_tasks import PixelwiseRegressionTask
from terratorch.tasks.object_detection_task import ObjectDetectionTask


@pytest.mark.parametrize(
    "task_type, expected_class",
    [
        ("classification", ClassificationTask),
        ("segmentation", SemanticSegmentationTask),
        ("multilabel_classification", MultiLabelClassificationTask),
        ("regression", PixelwiseRegressionTask),
        ("object_detection", ObjectDetectionTask),
    ],
)
def test_get_class_from_enum(task_type: str, expected_class: TerraTorchTask):
    t = TaskTypeEnum(value=task_type)
    type_class = t.get_class_from_enum()
    assert type(type_class) is type(expected_class)
