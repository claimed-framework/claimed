"""
# TODO: Update description
Tekton pipeline for with the following steps:
1. Step 1
"""

# TODO: Install kfp
# pip install kfp
# pip install kfp-tekton

import kfp
import kfp.dsl as dsl
import kfp.components as comp
from kfp_tekton.compiler import TektonCompiler

# TODO: Add your pipeline components based on the kfp yaml file from CLAIMED
# initialize operator from yaml file
component_op = comp.load_component_from_file('<operator>.yaml')
# initialize operator from remote file
web_op = comp.load_component_from_url('https://raw.githubusercontent.com/claimed-framework/component-library/main/component-library/<operator>.yaml')


# TODO: Update pipeline description, function name, and parameters
pipeline_name = 'my_pipeline'
# Pipeline function
@dsl.pipeline(
    name=pipeline_name,
    description="Pipeline description"
)
def my_pipeline(
    parameter1: str = "default_value",
    parameter2: str = "default_value",
):
    # TODO: Add the components and the required parameters
    step1 = component_op(
        parameter1=parameter1,
    )
    step2 = web_op(
        parameter2=parameter2,
    )

    # TODO: You can call multiple steps and created the dependencies
    step2.after(step1)

# TODO: Update pipeline function
# Kubernetes
kfp.compiler.Compiler().compile(pipeline_func=my_pipeline, package_path=f'{pipeline_name}.yaml')
# OpenShift with Tekton
TektonCompiler().compile(my_pipeline, f'{pipeline_name}.yaml')

print(f'Saved pipeline in {pipeline_name}.yaml')

# TODO: Run script with python
# TODO: Upload the yaml to KubeFlow
