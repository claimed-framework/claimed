from pathlib import Path
from typing import Any
import yaml
import pandas as pd
import click
from dataclasses import asdict
from benchmark.benchmark_types import (
    IterateBaseDataModule,
    Task,
    TaskTypeEnum,
)

GEOBENCH_TEMPLATE = Path(__file__).parent / 'geobenchv2_template.yaml'
PRITHVI_600M = 'prithvi_600M'


def _build_dataframe(config_files) -> pd.DataFrame:
    """
    build a pandas dataframe using the parameters of the specified config files
    """
    files = list()
    dataset = list()
    model = list()
    for config_file in config_files:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            try:
                backbone = config["model"]["init_args"]["model_args"]["backbone"]
                model.append(backbone)
                ds = config["data"]["init_args"]["cls"]
                dataset.append(ds)
                files.append(str(config_file))
            except KeyError as e:
                print(f"Error in file: {config_file}\n{e}")
                raise e
    return pd.DataFrame(data={"file": files, "model": model, "dataset": dataset})


def _create_basemodule(
    data: dict[str, Any], model_filter: str
) -> IterateBaseDataModule:
    """instantiate IterateBaseDataModule class based on the "data" field of the terratorch config

    Args:
        data (dict[str, Any]): _description_
        model_filter (str): model name is used to specify batch_size and eval_batch_size

    Returns:
        IterateBaseDataModule: subclass of torchgeo BaseDataModule that is part of iterate's config
    """
    dataset_class = data["class_path"]
    if "dict_kwargs" in data.keys():
        dict_kwargs = data["dict_kwargs"]
        dict_kwargs['batch_size'] = 8 if model_filter != PRITHVI_600M else 4
        dict_kwargs['eval_batch_size'] = 8 if model_filter != PRITHVI_600M else 4
        return IterateBaseDataModule(dataset_class=dataset_class, **dict_kwargs)
    else:
        return IterateBaseDataModule(dataset_class=dataset_class)


def _create_task(
    name: str,
    datamodule: IterateBaseDataModule,
    metric: str,
    terratorch_task: dict,
    task_type: TaskTypeEnum,
    direction: str,
    optimization_except: set[str] = set(),
    max_run_duration: str | None = None,
    early_stop_patience: int | None = None,
    early_prune: bool = False,
) -> dict:
    """instantiate Task dataclass and convert it to dict

    Args:
        name (str): name of the task - comes from terratorch config - data.init_args.cls
        datamodule (IterateBaseDataModule): _description_
        metric (str): _description_
        terratorch_task (dict): _description_
        task_type (TaskTypeEnum): type of task, e.g., regression, classification
        direction (str): direction to optimize
        optimization_except (set[str], optional): _description_. Defaults to set().
        max_run_duration (str | None, optional): _description_. Defaults to None.
        early_stop_patience (int | None, optional): _description_. Defaults to None.
        early_prune (bool, optional): _description_. Defaults to False.

    Returns:
        dict: _description_
    """
    task = Task(
        name=name,
        datamodule=datamodule,
        metric=metric,
        terratorch_task=terratorch_task,
        type=task_type,
        direction=direction,
        optimization_except=optimization_except,
        max_run_duration=max_run_duration,
        early_stop_patience=early_stop_patience,
        early_prune=early_prune,
    )
    task_dict = asdict(task)
    task_dict["type"] = task_type.value
    task_dict["datamodule"] = datamodule.to_dict()
    if len(optimization_except) == 0:
        del task_dict["optimization_except"]

    return task_dict


def _get_task_type(template: dict) -> TaskTypeEnum:
    tasks = template["tasks"]
    task = tasks[0]
    task_type = task["type"]
    assert isinstance(task_type, str)

    return TaskTypeEnum(value=task_type)


def _get_task_direction(template: dict) -> str:
    """extract task direction from template

    Args:
        template (dict): template created by user

    Returns:
        str: direction of the optimization (max or min)
    """
    tasks = template["tasks"]
    task = tasks[0]
    direction = task["direction"]
    assert isinstance(direction, str)
    assert direction in ["min", "max"]
    return direction


def _generate_iterate_config(
    directory: Path,
    output: Path,
    template: Path,
):

    config_files = directory.glob('**/*.yaml')
    files_df = _build_dataframe(config_files=config_files)

    files_df = files_df[files_df['dataset'].values != 'M4SAR']
    files_df = files_df[files_df['model'].values != 'resnet50_torchgeo']

    files_df = files_df.sort_values(['model', 'dataset'])

    models = files_df['model'].unique()

    with open(GEOBENCH_TEMPLATE, 'r') as file:
        template = yaml.safe_load(file)

    for model in models:

        tmp_df = files_df[files_df['model'].values == model]

        tasks = list()

        for i in range(tmp_df.shape[0]):

            with open(tmp_df['file'].values[i], 'r') as file:
                data = yaml.safe_load(file)

            name = tmp_df['dataset'].values[i]

            model_args: dict = data['model']['init_args']['model_args']
            # framework is an optional field of terratorch config
            if (
                model_args.get("framework") is not None
                and model_args.get("framework") == "faster-rcnn"
            ):
                metric = 'val_map'
            else:
                metric = 'val_segm_map'

            terratorch_task = data['model']['init_args']

            data = data['data']
            datamodule = _create_basemodule(data=data, model_filter=model)
            task_type = _get_task_type(template=template)
            task_direction = _get_task_direction(template=template)
            task = _create_task(
                name=name,
                datamodule=datamodule,
                metric=metric,
                terratorch_task=terratorch_task,
                task_type=task_type,
                direction=task_direction,
            )
            tasks.append(task)

    template['tasks'] = tasks

    with open(output, 'w') as file:
        yaml.dump(template, file)


@click.command()
@click.option(
    '--directory',
    prompt='Full path to the directory that contains all terratorch config yaml files',
    help='Full path to the directory that contains all terratorch config yaml files',
)
@click.option(
    '--output',
    prompt='Name of the config file that will be generated',
    help='Name of the config file that will be generated',
)
@click.option(
    '--template',
    prompt='Full path to the template file',
    help='Full path to the template file',
)
def generate_tt_iterate_config(directory: str, output: str, template: str):
    directory_path = Path(directory)
    assert directory_path.exists()
    template_path = Path(template)
    assert template_path.exists()
    output_path = Path(".") / output
    if output_path.exists():
        print(f"Delete existing {output_path} file")
        output_path.unlink()
    _generate_iterate_config(
        directory=directory_path, output=output_path, template=template_path
    )


if __name__ == '__main__':
    generate_tt_iterate_config()
