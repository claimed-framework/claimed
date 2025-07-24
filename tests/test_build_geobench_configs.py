from pathlib import Path
import pytest
import yaml
from benchmark.config_util.build_geobench_configs import generate_iterate_config
from deepdiff import DeepDiff


@pytest.mark.parametrize(
    "input_dir, output_dir, template, prefix",
    [
        (
            "/Users/ltizzei/Projects/Orgs/IBM/terratorch/examples/confs/geobenchv2_detection",
            "/Users/ltizzei/Projects/Orgs/IBM/terratorch-iterate/benchmark/config_util/",
            "/Users/ltizzei/Projects/Orgs/IBM/terratorch-iterate/benchmark/config_util/geobenchv2_template.yaml",
            "test_examples_confs_geobenchv2_detection",
        ),
        (
            "/Users/ltizzei/Projects/Orgs/IBM/terratorch-iterate/tests/test_config_util",
            "/Users/ltizzei/Projects/Orgs/IBM/terratorch-iterate/tests/test_config_util",
            "/Users/ltizzei/Projects/Orgs/IBM/terratorch-iterate/benchmark/config_util/geobenchv2_template.yaml",
            "test_config_util_",
        ),
    ],
)
def test__generate_iterate_config(input_dir, output_dir, template, prefix):
    input_dir_path = Path(input_dir)
    assert input_dir_path.exists()
    assert input_dir_path.is_dir()
    output_path = Path(output_dir)
    assert output_path.exists()
    assert output_path.is_dir()

    generate_iterate_config(
        input_dir=input_dir_path, output_dir=output_path, template=template, prefix=prefix
    )

    assert output_path.exists()

    oracle_config_files = [
        f for f in output_path.glob(f'**/geobench*.yaml') if "template" not in str(f)
    ]
    generated_config_files = output_path.glob(f'**/{prefix}*.yaml')
    for gen_config_file in generated_config_files:
        end_gen_config_filename = gen_config_file.name.replace(prefix, "")
        for oracle_config_file in oracle_config_files:
            end_oracle_config_filename = oracle_config_file.name.replace(
                "geobenchv2", ""
            )
            if end_gen_config_filename == end_oracle_config_filename:
                with open(gen_config_file, "r") as gen_file:
                    new_config = yaml.safe_load(gen_file)
                with open(oracle_config_file, "r") as gt_file:
                    oracle_config = yaml.safe_load(gt_file)

                oracle_tasks = oracle_config["tasks"]
                new_config_tasks = new_config["tasks"]
                # comparing the tasks
                for oracle_task in oracle_tasks:
                    oracle_task_name = oracle_task["name"]
                    found = False
                    for new_config_task in new_config_tasks:
                        new_config_task_name = new_config_task["name"]
                        if new_config_task_name == oracle_task_name:

                            diff = DeepDiff(new_config_task, oracle_task)
                            if len(diff) == 0:
                                found = True
                            else:
                                for k in [
                                    "datamodule",
                                    "direction",
                                    "metric",
                                    "terratorch_task",
                                    "type",
                                ]:
                                    diff = DeepDiff(new_config_task[k], oracle_task[k])
                                    assert len(diff) == 0, f"Error! {diff}"
                                found = True
                    assert found
