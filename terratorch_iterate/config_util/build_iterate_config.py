from pathlib import Path
import yaml
import pandas as pd
import click
from terratorch_iterate.iterate_types import (
    TaskTypeEnum,
)
from copy import deepcopy

DEFAULT_TEMPLATE = (
    Path(__file__).parent.parent.parent / "configs/templates/template.yaml"
)


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
            ds = str(config_file).split("/")[-1].split("_")[0]
            dataset.append(ds)
            # append file path
            files.append(str(config_file))
        except KeyError as e:
            msg = f"Error in file: {config_file}\n{e}"
            print(msg)
            raise KeyError(msg)

    df = pd.DataFrame(data={"file": files, "dataset": dataset})
    models = [
        x.split("/")[-1].replace(y + "_", "").replace(".yaml", "")
        for x, y in zip(df["file"].values, df["dataset"].values)
    ]
    df["model"] = models
    return df


def _create_task(
    name: str,
    datamodule: dict,
    metric: str,
    terratorch_task: dict,
    task_type: TaskTypeEnum,
    direction: str,
    max_run_duration: str | None = None,
    early_stop_patience: int | None = None,
    early_prune: bool | None = None,
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
    }
    # set optional fields if they are not None
    for k, v in [
        ("max_run_duration", max_run_duration),
        ("early_stop_patience", early_stop_patience),
        ("early_prune", early_prune),
    ]:
        if v is not None:
            task_dict[k] = v

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
    input: Path,
    output: Path,
    template: Path = DEFAULT_TEMPLATE,
    prefix: str = "tt-iterate-",
):
    """generate the tt-iterate based on yaml files located within the specified directory, based
    on previously defined template and save the result using specified output filename

    Args:
        input_dir (Path): contains all terratorch yaml files
        output_dir (Path): filename of the result
        template (Path): template file that contains pre-defined values
        prefix (str): prefix for creating new config files
    """
    assert input.exists()
    if input.is_dir():
        config_files = input.glob("**/*.yaml")
    elif input.is_file():
        config_files = [input]
    else:
        ValueError(f"Error! {input=} is neither a file nor a directory")
    files_df = _build_dataframe(config_files=config_files)

    # set default values if necessary
    if template is None:
        template = DEFAULT_TEMPLATE
    if prefix is None:
        prefix = "tt-iterate-"

    models = files_df["model"].unique()

    with open(template, "r") as file:
        template_dict: dict = yaml.safe_load(file)

    # generate one config per model
    for model in models:
        model_specific_template = deepcopy(template_dict)
        # create unique name for experiment
        model_specific_template["experiment_name"] = f"{prefix}_{model}"
        tasks = list()

        # filter dataframe by model
        single_model_df = files_df[files_df["model"].values == model]

        for i in range(single_model_df.shape[0]):
            # open terratorch config file
            with open(single_model_df["file"].values[i], "r") as file:
                data = yaml.safe_load(file)

            name = single_model_df["dataset"].values[i]

            model_args: dict = data["model"]["init_args"]["model_args"]
            # framework is an optional field of terratorch config
            if (
                model_args.get("framework") is not None
                and model_args.get("framework") == "faster-rcnn"
            ):
                metric = "val_map"
            else:
                metric = "val/loss"

            # terratorchtask is extracted from the data.model.init_args of terratorch config file
            terratorch_task = data["model"]["init_args"]
            # create datamodule based on data field
            datamodule = data["data"]
            task_type = _get_task_type(template=template_dict)
            task_direction = _get_task_direction(template=template_dict)
            task = _create_task(
                name=name,
                datamodule=datamodule,
                metric=metric,
                terratorch_task=terratorch_task,
                task_type=task_type,
                direction=task_direction,
            )
            tasks.append(task)

        model_specific_template["tasks"] = tasks
        if output.is_dir():
            path = output / f"{prefix}_{model}.yaml"
        else:
            path = output
        if path.exists():
            path.unlink()
        with open(path, "w") as file:
            yaml.dump(model_specific_template, file)
            print(f"{path} file has been created")


@click.command()
@click.option(
    "--input_dir",
    prompt="Full path to the directory that contains all terratorch config yaml files",
    help="Full path to the directory that contains all terratorch config yaml files",
)
@click.option(
    "--output_dir",
    prompt="Full path to the directory in which the new config files will be stored",
    help="Full path to the directory in which the new config files will be stored",
)
@click.option(
    "--template",
    prompt="Full path to the template file",
    help="Full path to the template file",
)
@click.option(
    "--prefix",
    prompt="Prefix of the config filename, e.g., my-config-",
    help="Prefix of the config filename",
)
def generate_tt_iterate_config(
    input_dir: str, output_dir: str, template: str, prefix: str
):
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
        input=directory_path,
        output=output_path,
        template=template_path,
        prefix=prefix,
    )


if __name__ == "__main__":
    generate_tt_iterate_config()
