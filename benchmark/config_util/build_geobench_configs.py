from pathlib import Path
from typing import Any
import yaml
import pandas as pd
import click
from benchmark.benchmark_types import (
    TaskTypeEnum,
)
from copy import deepcopy

PRITHVI_600M = 'prithvi_600M'


def _build_dataframe(config_files) -> pd.DataFrame:
    """
    build a pandas dataframe using the parameters of the specified config files
    """
    files = list()
    dataset = list()
    models = list()
    for config_file in config_files:
        try:
            # extract dataset name from filename
            ds = str(config_file).split('/')[-1].split('_')[0]
            dataset.append(ds)
            # append file path
            files.append(str(config_file))
        except KeyError as e:
            msg = f"Error in file: {config_file}\n{e}"
            print(msg)
            raise KeyError(msg)

    df = pd.DataFrame(data={"file": files, "dataset": dataset})
    models = [
        x.split('/')[-1].replace(y + '_', '').replace('.yaml', '')
        for x, y in zip(df['file'].values, df['dataset'].values)
    ]
    df["model"] = models
    return df


def _create_basemodule(data: dict[str, Any], model_filter: str) -> dict:
    """create a dict based on the "data" field of the terratorch config

    Args:
        data (dict[str, Any]): _description_
        model_filter (str): model name is used to specify batch_size and eval_batch_size

    Returns:
        dict: returns a dict that represents the datamodule field of iterate config file
    """
    base_module = dict()
    base_module["class_path"] = data["class_path"]
    if "dict_kwargs" in data.keys():
        dict_kwargs = data["dict_kwargs"]
        batch_size = 8 if model_filter != PRITHVI_600M else 4
        dict_kwargs["batch_size"] = batch_size
        dict_kwargs['eval_batch_size'] = 8 if model_filter != PRITHVI_600M else 4

        base_module["dict_kwargs"] = dict_kwargs
    base_module["init_args"] = data["init_args"]
    return base_module


def _create_task(
    name: str,
    datamodule: dict,
    metric: str,
    terratorch_task: dict,
    task_type: TaskTypeEnum,
    direction: str,
    max_run_duration: str | None = None,
    early_stop_patience: int | None = None,
    early_prune: bool = False,
) -> dict:
    """instantiate Task dataclass and convert it to dict

    Args:
        name (str): name of the task - comes from terratorch config - data.init_args.cls
        datamodule (dict): _description_
        metric (str): _description_
        terratorch_task (dict): _description_
        task_type (TaskTypeEnum): type of task, e.g., regression, classification
        direction (str): direction to optimize
        max_run_duration (str | None, optional): _description_. Defaults to None.
        early_stop_patience (int | None, optional): _description_. Defaults to None.
        early_prune (bool, optional): _description_. Defaults to False.

    Returns:
        dict: _description_
    """

    task_dict = {
        "name": name,
        "datamodule": datamodule,
        "type": task_type.value,
        "direction": direction,
        "metric": metric,
        "terratorch_task": terratorch_task,
        "max_run_duration": max_run_duration,
        "early_stop_patience": early_stop_patience,
        "early_prune": early_prune,
    }

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


def generate_iterate_config(
    input_dir: Path, template: Path, output_dir: Path, prefix: str = "test_"
):
    """generate the tt-iterate based on yaml files located within the specified directory, based
    on previously defined template and save the result using specified output filename

    Args:
        input_dir (Path): contains all terratorch yaml files
        output_dir (Path): filename of the result
        template (Path): template file that contains pre-defined values
    """

    config_files = input_dir.glob('**/*.yaml')
    files_df = _build_dataframe(config_files=config_files)

    files_df = files_df[files_df['dataset'].values != 'M4SAR']
    files_df = files_df[files_df['model'].values != 'resnet50_torchgeo']

    files_df = files_df.sort_values(['model', 'dataset'])

    models = files_df['model'].unique()

    with open(template, 'r') as file:
        template = yaml.safe_load(file)

    for model in models:
        model_specific_template = deepcopy(template)
        model_specific_template["experiment_name"] = f"{prefix}_{model}"
        tasks = list()

        single_model_df = files_df[files_df['model'].values == model]

        for i in range(single_model_df.shape[0]):

            with open(single_model_df['file'].values[i], 'r') as file:
                data = yaml.safe_load(file)

            name = single_model_df['dataset'].values[i]

            model_args: dict = data['model']['init_args']['model_args']
            # framework is an optional field of terratorch config
            if (
                model_args.get("framework") is not None
                and model_args.get("framework") == "faster-rcnn"
            ):
                metric = 'val_map'
            else:
                metric = 'val_segm_map'

            # terratorchtask is the data.model.init_args of terratorch config file
            terratorch_task = data['model']['init_args']
            # create datamodule based on data field
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

        model_specific_template['tasks'] = tasks
        path = output_dir / f"{prefix}_{model}.yaml"
        if path.exists():
            path.unlink()
        with open(path, 'w') as file:
            yaml.dump(model_specific_template, file)
            print(f"{path} file has been created")


@click.command()
@click.option(
    '--input_dir',
    prompt='Full path to the directory that contains all terratorch config yaml files',
    help='Full path to the directory that contains all terratorch config yaml files',
)
@click.option(
    '--output_dir',
    prompt='Full path to the directory in which the new config files will be stored',
    help='Full path to the directory in which the new config files will be stored',
)
@click.option(
    '--template',
    prompt='Full path to the template file',
    help='Full path to the template file',
)
@click.option(
    '--prefix',
    prompt='Prefix of the config filename, e.g., my-config-',
    help='Prefix of the config filename',
)
def generate_tt_iterate_config(input_dir: str, output_dir: str, template: str, prefix: str):
    directory_path = Path(input_dir)
    assert directory_path.exists()
    assert directory_path.is_dir
    
    template_path = Path(template)
    assert template_path.exists()
    assert template_path.is_file
    
    output_path = Path(output_dir)
    assert output_path.exists()
    assert output_path.is_dir

    assert isinstance(prefix, str), f"Error! {type(prefix)} is not a str"
    generate_iterate_config(
        input_dir=directory_path,
        output_dir=output_path,
        template=template_path,
        prefix=prefix,
    )


if __name__ == '__main__':
    generate_tt_iterate_config()
