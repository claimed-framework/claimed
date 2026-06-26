from pathlib import Path

from jsonargparse import ArgumentParser, Namespace
from terratorch_iterate.iterate_types import Task
import uuid
import pytest


@pytest.mark.skip()
def test_launch_training():
    # experiment_name='dofa_large_patch16_224_upernetdecoder_true_modified_continue_False_test_models_True' metric='val/loss' storage_uri='/dccstor/geofm-finetuning/terratorch-iterate-test/39d14a9ed79e4ee39739fa92a4cdd758/hpo' direction='max'
    random_hex = uuid.uuid4().hex

    storage_uri = Path(f"/tmp/{random_hex}")
    if not storage_uri.exists():
        storage_uri.mkdir()
    parser = ArgumentParser()
    config_path = (
        Path(__file__).parent.parent.parent
        / "configs/tests/dofa_large_patch16_224_upernetdecoder_true_modified.yaml"
    )
    assert config_path.exists()
    config = parser.parse_path(config_path)
    config_init: Namespace = parser.instantiate_classes(config)
    tasks = config_init.tasks
    assert isinstance(tasks, list), f"Error! {tasks=} is not a list"
    for t in tasks:
        assert isinstance(t, Task), f"Error! {t=} is not a Task"
    # data_module = MNzCattleNonGeoDataModule()
    # trainer = Trainer(**training_spec_copy.trainer_args)
    # launch_training(
    #     trainer=trainer,
    #     datamodule=datamodule,
    #     experiment_name=experiment_name,
    #     metric=metric,
    #     direction=direction,
    #     storage_uri=storage_uri,
    # )
