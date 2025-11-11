# Getting Started with CLAIMED

The [CLAIMED framework](https://github.com/claimed-framework) enables ease-of-use development and deployment of cloud native data processing applications on Kubernetes using operators and workflows.  

A central tool of CLAIMED is the **Claimed Component Compiler (C3)** which creates a docker image with all dependencies, pushes the container to a registry, and creates a kubernetes-job.yaml as well as a kubeflow-pipeline-component.yaml. 
This page explains how to apply operators, combine them to workflows, and how to build them yourself using C3.

If you like CLAIMED, just give us a [star](https://github.com/claimed-framework/component-library) on our [main project](https://github.com/claimed-framework/component-library).


## Content

**[1. Apply operators](#1-apply-operators)**

**[2. Operator library](#2-operator-library)**

**[3. Create workflows](#3-create-workflows)**

**[4. Create operators](#4-create-operators)**

**[5. Create grid wrapper](#5-create-grid-wrapper)**

---

## 1. Apply operators

An operator is a single processing step. You can run the script locally with the [CLAIMED CLI](https://github.com/claimed-framework/cli) using the following command:
```shell
claimed --component <registry>/<image>:<version> --<parameter1> <value2> --<parameter2> <value2> ...
```

Besides CLAIMED CLI, you can use an operator in [workflows](#3-create-workflows), or deploy a kubernetes job using the `job.yaml` which is explained in the following.


### 1.1 Specify the job

First, update the variable values in the `job.yaml`. 
You can delete a variable to use the default value, if one is defined. 
The default values are listed in the KubeFlow component `yaml` file.

#### Secrets

You can use key-value secrets for passing credentials to the job. Save the secrets to the cluster and replace the `value: ...` with the following pattern in the `job.yaml`:  

```yaml
      containers:
        env:
        - name: <variable>
          valueFrom:
            secretKeyRef:
              name: <secret-name>
              key: <secret_key>

# Example for an access key
      containers:
        env:
        - name: access_key_id
          valueFrom:
            secretKeyRef:
              name: cos-secret
              key: access_key_id
```

#### Container registry

If the container image is saved in a non-public registry, add an image pull secret to the container specs. Check `image: ...` in the `job.yaml` to find the location of the container image. If it includes a non-public registry like [icr.io](), you need to provide the image pull secret at the end of the file:  

```yaml
    spec:
      containers:
      - name: example-script
        image: icr.io/namespace/claimed-example-script:0.1
        ...:
      imagePullSecrets:
        - name: <pull-secret-name>
```

#### Storage

You can provide access to a Kubernetes/OpenShift persistent volume by specifying it in the `job.yaml`. 
OpenShift clusters require specifying the security context on the pod/template spec level. 
You get the group ID for the volume from your administrator. 
You can use `/opt/app-root/src/<mount_dir>` to mount the `mount_dir` in the working directory of the pod.  

```yaml
    spec:
      containers:
        ...:
        volumeMounts:
          - name: data
            mountPath: /opt/app-root/src/<mount_dir>
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
              - ALL
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: <persistent-volume-name>
      securityContext:
        supplementalGroups: [<group ID>]      
```

#### Error handling

If a pod fails, it is restarted by the job until it finishes successfully. You can specify the error handling in the `job.yaml`.
First, `backoffLimit` limits the number of restarts (default: 5). Second, `restartPolicy` defines if a failed pod restarts (`OnFailure`) or if a new pod is created while the failed pod stops with the error (`Never`).

```yaml
spec:
  backoffLimit: 1
  template:
    spec:
      ...:
      restartPolicy: Never
```

#### Example

The following is an exemplary `example_script.job.yaml` that includes a `imagePullSecret` and mounts a persistent volume claim from a cluster. 
Variables that are not defined are using the default value.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: example-script
spec:
  template:
    spec:
      containers:
      - name: example-script
        image: docker.io/user/claimed-example-script:0.1
        command: ["/opt/app-root/bin/ipython","/opt/app-root/src/example_script.py"]
        env:
        - name: input_path
          value: "data/"
        - name: num_values
          value: "5"
        volumeMounts:
          - name: pvc-data
            mountPath: /opt/app-root/src/data/
      volumes:
        - name: pvc-data
          persistentVolumeClaim:
            claimName: pvc-name
      restartPolicy: OnFailure
      imagePullSecrets:
        - name: user-pull-secret
```


### 1.2 Cluster CLI login

You can start jobs with the `kubectl` (Kubernetes) or `oc` (OpenShift) CLI. If your using Kubernetes, the login procedure includes multiple steps which are detailed in the [Kubernetes docs](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/).

Logging into an OpenShift cluster is easier. You can use a token which you can generate via the browser UI, or you're username. You might want to add `--insecure-skip-tls-verify` when errors occur.

```sh
# Login via token (Browser login > Your name > Copy login command > Display token)
oc login --token=<token> --server=<server_url> --insecure-skip-tls-verify

# Login via user name
oc login <server_url> -u <user>

# Optional: Change default project
oc project <project>
```

### 1.3 Start and manage jobs

After specifying the `job.yaml` and logging into the cluster, you can start or stop a job via the CLI. If your using an OpenShift cluster, you simply replace `kubectl` with `oc` in the commands.

```sh
# start job
kubectl apply -f <operator>.job.yaml 

# kill job
kubectl delete -f <operator>.job.yaml
```
Note that calling `kubectl apply` two times can lead to an error because jobs have unique names. If a job with the same name is running, you might need to kill the job before restarting it.

The job creates a pod which is accessible via the browser UI or via CLI using the standard kubectl commands.
```sh
# list all pods in the current project
kubectl pods

# get logs of a pod
kubectl logs -f <pod-name>

# pod description
kubectl describe pod <pod-name>
```

---

## 2. Operator library

Reusable code is a key idea of CLAIMED and operator libraries make it easier to share single processing steps. 
Because each operator includes a docker image with specified dependencies, operators can be easily reused in different workflows. 

Public operators are accessible from the [CLAIMED component library](https://github.com/claimed-framework/component-library/tree/main/component-library). 

You can run a public operator locally by using [claimed-cli](https://github.com/claimed-framework/cli) or copy the Kubernetes job.yaml file for running the operator on a Kubernetes/OpenShift cluster. 
You can also use the operators in workflows as explained in the next section.     

---

## 3. Create workflows

Multiple operators can be combined to a workflow, e.g., a KubeFlow pipeline or a CWL workflow. Therefore, C3 creates `<operator>.yaml` files which define a KFP component and `<operator>.cwl` files for a CWL step. 

### KubeFlow Pipeline

After initializing your operators, you can combine them in a pipeline function: 

```python
# pip install kfp

import kfp.components as comp
import kfp
import kfp.dsl as dsl

# initialize operator from yaml file
file_op = comp.load_component_from_file('<operator>.yaml')
# initialize operator from remote file
web_op = comp.load_component_from_url('https://raw.githubusercontent.com/claimed-framework/component-library/main/component-library/<operator>.yaml')

@dsl.pipeline(
    name="my_pipeline",
    description="Description",
):
def my_pipeline(
    parameter1: str = "value",
    parameter2: int = 1,
    parameter3: str = "value",
):
    step1 = file_op(
        parameter1=parameter1,
        parameter2=parameter2,
    )

    step2 = web_op(
        parameter1=parameter1,
        parameter3=parameter3,
    )
    
    step2.after(step1)

kfp.compiler.Compiler().compile(pipeline_func=my_pipeline, package_path='my_pipeline.yaml')
```

When running the script, the KFP compiler generates a `<pipeline>.yaml` file which can be uploaded to the KubeFlow UI to start the pipeline.
Alternatively, you can run the pipeline with the SDK client, see [KubeFlow Docs](https://www.kubeflow.org/docs/components/pipelines/v1/sdk/build-pipeline/) for details.

If your using an OpenShift cluster, your might want to use the Tekton compiler.

```python
# pip install kfp-tekton

from kfp_tekton.compiler import TektonCompiler

TektonCompiler().compile(pipeline_func=my_pipeline, package_path='my_pipeline.yaml')
```

If you are using another tekton version, you can use the following code to save an adjusted yaml file for version `v1beta1`: 

```python
# pip install kfp-tekton pyyaml

import yaml
from kfp_tekton.compiler import TektonCompiler

# Read dict to update apiVersion
_, pipeline_dict = TektonCompiler().prepare_workflow(my_pipeline)
pipeline_dict['apiVersion'] = 'tekton.dev/v1beta1'
# write pipeline to yaml
with open('my_pipeline.yaml', 'w') as f:
    yaml.dump(pipeline_dict, f)
```

#### Timeout in KubeFlow Tekton

The default timeout in a KFP tekton pipeline is set to 60 minutes. The default value can be changed in the tekton config by the [administrators](https://tekton.dev/docs/pipelines/pipelineruns/#configuring-a-failure-timeout). Otherwise, you can update the timeout in the yaml with the following code:

```python
# Read dict to update apiVersion and timeouts
_, pipeline_dict = TektonCompiler().prepare_workflow(my_pipeline)
pipeline_dict['spec']['timeouts'] = {'pipeline': "0"}  # 0 = no timeout
# write pipeline to yaml
with open('my_pipeline.yaml', 'w') as f:
    yaml.dump(pipeline_dict, f)
```

#### Shared volumes

Data is not shared by default between different steps. 
You can add a volume to each step for data sharing. 
First, you create a PersistentVolumeClaim (PVC) in the Kubernetes project that is running KubeFlow.
If you want to run multiple steps in parallel, this PVC must support ReadWriteMany, otherwise ReadWirteOnce is sufficient.
Next, you can mount this PVC to each step with the following code:

```python
mount_folder = "/opt/app-root/src/<folder>"

# Init the KFP component
step = my_kfp_op(...)

step.add_pvolumes({mount_folder: dsl.PipelineVolume(pvc='<pvc_name>')})
```

You can include the working directory in the mount path to use relative paths (`/opt/app-root/src/` for python and `home/docker` for R). 
Otherwise, you can use absolute paths in your scripts/variables `/<folder>/...`. 

#### Secrets

You can use key-value secrets in KubeFlow as well to avoid publishing sensible information in pod configs and logs. 
You can add the secrets in the Kubernetes project that is running KubeFlow.
Then, you can add secrets to a specfic step in the pipeline with the following code:

```python
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1SecretKeySelector

# Init the KFP component
step = my_kfp_op(...)

# Add a secret as env variable
secret_env_var = V1EnvVar(
    name='<variable_name>',
    value_from=V1EnvVarSource(secret_key_ref=V1SecretKeySelector(name='<secret_name>', key='<secret_key>')
))
step.add_env_variable(secret_env_var)
```

The secret will be set as an env variable and load by the common C3 interface.
Therefore, it is important that KubeFlow does not everwrite this env variable. 
You need to adjust the command in the KFP component yaml by deleting the variable: 
```yaml
# Original command with secret_variable
command:
  ...
  python ./<my_script>.py log_level="${0}" <secret_variable>="${1}" other_variable="${2}" ...
  ...

# Adjusted command
command:
  ...
  python ./<my_script>.py log_level="${0}" other_variable="${2}" ...
  ...
```
Further, it is important, that the variable has a default value and is optional 
(You can simply add `default: ""` to the variable in the KFP component yaml without recompiling your script).


### CWL workflows

You can run workflows locally with CWL. This requires the cwltool package:
```shell
pip install cwltool
```

You can create a CWL workflow by combining multiple CWL steps:

```text
cwlVersion: v1.0
class: Workflow

inputs:
  parameter1: string
  parameter2: string
  parameter3: string
  parameter4: string
outputs: []

steps:
  <component>.cwl:
    run: ./path/to/<component>.cwl
    in:
        parameter1: parameter1
        parameter2: parameter2
        parameter3: parameter3
    out: []
  <component2>.cwl:
    run: ./path/to/<component2>.cwl
    in:
        parameter3: parameter3
        parameter4: parameter4
    out: []
``` 

Run the CWL workflow in your terminal with:
```shell
cwltool <workflow>.cwl --parameter1 <value1> --parameter2 <value2> --parameter3 <value3> --parameter4 <value4>
```

---

## 4. Create operators

### 4.1 Download C3

You can install C3 via pip:
```sh
pip install claimed
```

### 4.2 C3 requirements

Your operator script has to follow certain requirements to be processed by C3. Currently supported are python scripts and ipython notebooks.

#### Python scripts

- The operator name is the python file: `my_operator_name.py` -> `claimed-my-operator-name`
- The operator description is the first doc string in the script: `"""Operator description"""`
- The required pip packages are listed in comments starting with pip install: `# pip install <package1> <package2>` or `# pip install -r ~/requierments.txt`
- The interface is defined by environment variables `my_parameter = os.getenv('my_parameter')`. 
- You can cast a specific type by wrapping `os.getenv()` with `int()`, `float()`, `bool()`. The default type is string. Only these four types are currently supported. You can use `None` as a default value but not pass the `NoneType` via the `job.yaml`.
- Output paths for KubeFlow can be defined with `os.environ['my_output_parameter'] = ...'`. Note that operators cannot return values but always have to save outputs in files.

You can optionally install future tools with `dnf` by adding a comment `# dnf <command>`. 

#### iPython notebooks

- The operator name is the notebook file: `my_operator_name.ipynb` -> `claimed-my-operator-name`
- The notebook is converted by `nbconvert` to a python script before creating the operator by merging all cells. 
- Markdown cells are converted into doc strings. shell commands with `!...` are converted into `get_ipython().run_line_magic()`.
- The requirements of python scripts apply to the notebook code (The operator description can be the first markdown cell).

#### R scripts

- The operator name is the python file: `my_operator_name.R` -> `claimed-my-operator-name`
- The operator description is currently fixed to `"R script"`.
- The required R packages are installed with: `install.packages(<packname>, repos=<optional repo>)`
- The interface is defined by environment variables `my_parameter <- Sys.getenv('my_parameter', 'optional_default_value')`. 
- You can cast a specific type by wrapping `Sys.getenv()` with `as.numeric()` or `as.logical()`. The default type is string. Only these three types are currently supported. You can use `NULL` as a default value but not pass `NULL` via the `job.yaml`.
- Output paths for KubeFlow can be defined with `Sys.setenv()`. Note that operators cannot return values but always have to save outputs in files.

You can optionally install future tools with `apt` by adding a comment `# apt <command>`.

#### Example

The following is an example python script `example_script.py` that can be compiled by C3.

```py
"""
This is the operator description. 
The file name becomes the operator name.
"""

# Add dependencies by comments starting with "pip install". 
# You can add multiple comments if the packages require a specific order.
# pip install numpy

import os
import logging
import numpy as np

# A comment one line above os.getenv is the description of this variable.
input_path = os.getenv('input_path')

# You can cast a specific type with int(), float(), or bool().
num_values = int(os.getenv('num_values', 5))

# Output paths are starting with "output_". 
output_path = os.getenv('output_path', None)


def my_function(n_random):
    """
    The compiler only includes the first doc string.This text is not included.
    """
    random_values = np.random.randn(n_random)
    # You can use logging in operators. 
    # C3 adds a logger and a parameter log_level (default: 'INFO') to the operator. 
    logging.info(f'Random values: {random_values}')

    
if __name__ == '__main__':
    my_function(num_values)

```

### 4.3 Docker engine
C3 requires a running Docker engine to build the container image. A popular app is [Docker Desktop](https://www.docker.com/products/docker-desktop/). However, Docker Desktop requires licences for commercial usage in companies. An open source alternatives is [Rancher Desktop](https://rancherdesktop.io) (macOS/Windows/Linux) which includes docker engine and a UI. A CLI alternative for macOS and Linux is [Colima](https://github.com/abiosoft/colima) which creates a Linux VM for docker. 

```sh
# Install Colima with homebrew
brew install docker docker-compose colima
 
# Start docker VM
colima start

#Stop docker VM
colima stop
```

### 4.4 Container registry

C3 creates a container image for the operator which has to be stored in a container registry. A simple solution for non-commercial usage is Docker Hub, but it has limited private usage. 
Alternative to a professional plan from Docker Hub are the [IBM Cloud registry](https://www.ibm.com/products/container-registry) or [Amazon ECR](https://aws.amazon.com/ecr/).

After starting the Docker engine, you need to login to the registry with docker.
 
```sh
docker login -u <user> -p <pw> <registry>/<namespace>
```

### 4.5 Compile an operator with C3

With a running Docker engine and your operator script matching the C3 requirements, you can execute the C3 compiler by running `create_operator.py`:

```sh
c3_create_operator --repository "<registry>/<namespace>" "<my-operator-script>.py" "<additional_file1>" "<additional_file2>"       
```

You need to provide the repository with `--repository` or `-r`. You can specify the version of the container image (default: "0.1") with `--version` or `-v`.
The first positional argument is the path to the python script or the ipython notebook. Optional, you can define additional files that are copied to the container images in the following positinal arguments. You can use wildcards for additional files. E.g., `*` would copy all files in the current directory to the container image. (Hidden files and directories must be specified. Be aware of `data/` folders and others before including all files.)
Note,that the docker build messages are suppressed by default. If you want to display the docker logs, you can add `--log_level DEBUG`.

View all arguments by running:
```sh
c3_create_operator --help
```

C3 generates the container image that is pushed to the registry, a `<my-operator-script>.yaml` file for KubeFlow, a `<my-operator-script>.job.yaml` for Kubernetes, and a `<my-operator-script>.cwl` file for CWL.

### 4.6 CLAIMED Containerless Operators
CLAIMED containerless operators allow you to execute scripts as fully functional workflow components without the need for traditional containerization.

After installing the claimed component compiler via pip install claimed c3, you can compile a script into a containerless operator just as you would for containerized components like Docker, Kubernetes (jobs, pods, deployments), Kubeflow, or Apache Airflow.

Using the command c3_create_containerless_operator my_script.py, your script is transformed into a standalone, executable operator. An example of a containerless operator can be found in the [containerless-bootstrap repository](https://github.com/claimed-framework/containerless-bootstrap). These operators can be executed seamlessly using the claimed CLI, replacing the container registry path with the containerless prefix. For instance, running claimed --component containerless/claimed-util-cos:latest --cos_connection cos://access_key_id:secret_access_key@s3.us-east.cloud-object-storage.appdomain.cloud/some_bucket/some_path --operation put --local_path some_file.zip enables cloud object storage operations with the 'claimed-util-cos' without requiring a container runtime. This approach significantly reduces overhead and speeds up execution while maintaining compatibility with established workflow orchestration frameworks.


---

## 5. Create grid wrapper

You can use grid computing to parallelize an operator. 
The grid computing requires that the code is parallelizable, e.g., by processing different files.
Therefore, the code gets wrapped by a coordinator script: The grid wrapper.

### 5.1 C3 grid computing requirements

You can use the same code for the grid wrapper as for an operator by adding an extra functon which is passed to C3. 
The grid wrapper executes this function in each batch and passes specific parameters to the function: 
The first parameter is the batch id, followed by all variables defined in the operator interface. 
You need to adapt the variables based on the batch, e.g., by adding the batch id to input and output paths.

```python
def grid_process(batch_id, parameter1, parameter2, *args, **kwargs):
    # update operator parameters based on batch id
    parameter1 = parameter1 + batch_id
    parameter2 = os.path.join(parameter2, batch_id)

    # execute operator code with adapted parameters 
    my_function(parameter1, parameter2)
```

You might want to add `*args, **kwargs` to avoid errors, if not all interface variables are used in the grid process.
Note that the operator script is imported by the grid wrapper script. Therefore, all code in the script is executed. 
If the script is also used as a single operator, it is recommended to check for `__main__` to avoid executions when the code is imported by the grid wrapper.

```python
if __name__ == '__main__':
    my_function(parameter1, parameter2)
```

Note that the grid computing is currently not implemented for R scripts. 

### 5.2 Compile a grid wrapper with C3

The compilation is similar to an operator. Additionally, the name of the grid process is passed to `create_gridwrapper.py` using `--process` or `-p` (default: `"grid_process"`) 
and a backend for the coordinator is selected with `--backend` or `-b` (default: `"local"`).    

```sh
c3_create_gridwrapper -r "<registry>/<namespace>" -p "grid_process" -b "local" "<my-operator-script>.py" "<additional_file1>" "<additional_file2>" 
```

C3 supports three backends for the coordination: Coordinator files on a shared local storage (`"local"`), on COS (`"cos"`), or as a key-value storage on S3 (`"s3kv"`).

Note, that the backend `"legacy_cos"` also handles downloading and uploading files from COS. We removed this functionality to simplify the grid wrapper.

The grid wrapper creates a temporary file `gw_<my-operator-script>.py` which is copied to the container image and deleted.  
Similar to an operator, `gw_<my-operator-script>.yaml`, `gw_<my-operator-script>.cwl`, and `gw_<my-operator-script>.job.yaml` are created.


### 5.3 Apply grid wrappers

The grid wrapper uses coordinator files to split up the batch processes between different pods. 
Therefore, each pod needs access to a shared persistent volume, see [storage](#storage).
Alternatively, you can use the COS or S3kv grid wrapper which uses a coordinator in S3.

The grid wrapper adds specific variables to the `job.yaml`, that define the batches and some coordination settings.

First, you can define the list of batch ids in a file and pass `gw_batch_file` to the grid wrapper. 
You can use either a `txt` file with a comma-separated list of strings, a `json` file with the keys being the batch ids, or a `csv` file with `gw_batch_file_col_name` being the column with the batch ids.
`gw_batch_file` can be a local path, a path within the coordinator bucket or a COS connection to a file (`cos://<access_key_id>:<access_secret_key>@<endpoint>/<bucket>/<path_to>/<batch_file>`).

Second, you need to define a `gw_coordinator_path` or `gw_coordinator_connection`.
The `gw_coordinator_path` is used in the `local` version. It is a path to a persistent and shared directory that is used by the pods to lock batches and mark them as processed.
`gw_coordinator_connection` is used in the `cos` and `s3kv` version. It defines a connection to a directory on COS: `cos://<access_key_id>:<access_secret_key>@<endpoint>/<bucket>/<path_to_directory>`.
The coordinator uses files with specific suffixes: `.lock`, `.processed`, and `.err`.
`gw_lock_timeout` defines the time in seconds until other pods remove the `.lock` file from batches that might be struggling (default `10800`). 
If your processes run very long, you can increase `gw_lock_timeout` to avoid duplicated processing of batches.
By default, pods skip batches with `.err` files. You can set `gw_ignore_error_files` to `True` after you fixed the error.

The grid wrapper currently does not support [secrets](#secrets) for the access key and secret within a connection.

Lastly, you want to add the number of parallel pods by adding `parallelism : <num pods>` to the `job.yaml`.

```yaml
spec:
  parallelism: 10
```

In KubeFlow pipelines, you can call the grid wrapper multiple times via a `for` loop. Note that the following step needs to wait for all parallel processes to finish.

```python
process_parallel_instances = 10

@dsl.pipeline(...)
def preprocessing_val_pipeline(...):    
    step1 = first_op()
    step3 = following_op()
    
    for i in range(process_parallel_instances):
        step2 = grid_wrapper_op(...)

        step2.after(step1)
        step3.after(step2)
```

#### Local example

The local grid wrapper requires a local storage for coordination like the PVC in the following example.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: cgw-my-operator
spec:
  parallelism: 10
  template:
    spec:
      containers:
      - name: cgw-my-operator
        image: us.icr.io/geodn/claimed-cgw-my-operator:0.01
        command: ["/opt/app-root/bin/python","/opt/app-root/src/claimed_cgw_my_operator.py"]
        env:
        - name: gw_batch_file
          value: "data/schedule.json"
        - name: gw_coordinator_path
          value: 'gw_coordinator'
        - name: my_operator_data_path
          value: 'data/*'
        - name: my_operator_target_path
          value: 'data/output/'
        - name: my_operator_parameter
          value: "100"
        volumeMounts:
          - name: pvc-data
            mountPath: /opt/app-root/src/data/
      volumes:
        - name: pvc-data
          persistentVolumeClaim:
            claimName: pvc-name
      restartPolicy: Never
      imagePullSecrets:
        - name: image-pull-secret
```

#### COS example

The COS grid wrapper uses a COS bucket for downloading and uploading batch data and coordination.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: cgw-my-operator
spec:
  parallelism: 10
  template:
    spec:
      containers:
      - name: cgw-my-operator
        image: us.icr.io/geodn/claimed-cgw-my-operator:0.01
        command: ["/opt/app-root/bin/python","/opt/app-root/src/claimed_cgw_my_operator.py"]
        env:
        - name: gw_file_path_pattern
          value: 'data/*'
        - name: gw_group_by
          value: '[-10:-4]'
        - name: gw_source_access_key_id
          valueFrom:
            secretKeyRef:
              name: cos-secret
              key: access_key_id
        - name: gw_source_secret_access_key
          valueFrom:
            secretKeyRef:
              name: cos-secret
              key: secret_access_key
        - name: gw_source_endpoint
          value: 'https://s3.cloud-object-storage.cloud'
        - name: gw_source_bucket
          value: 'my-bucket'
        - name: gw_target_path
          value: 'cos_results'
        - name: gw_coordinator_path
          value: 'gw_coordinator'
        - name: my_operator_data_path
          value: 'input'
        - name: my_operator_target_path
          value: 'target'
        - name: my_operator_parameter
          value: "100"
      restartPolicy: Never
      imagePullSecrets:
        - name: image-pull-secret
```

### 5.4 Simple Grid Wrapper
Although CLAIMED grid wrappers with the different coordinator plugins are very powerful, sometimes it is also overwhelming. Therefore we created the simple_grid_wrapper plugin which allows you to just point as many parallel workers as you like to a directory of files. Those files are randomly processed by each worker, making sure there is only one worker processing a file. Once all files are processed, the results are renamed to original_file_name.PROCESSED.ext. Please have a look at the examples folder to create your own simple grid wrapper. Here are the commands, given you are in the examples folder of this repository:

```
(pip install claimed c3)  
c3_create_gridwrapper simple_grid_wrapper_example.py -b simple_grid_wrapper 
export CLAIMED_DATA_PATH=/path/to/your/c3/examples
claimed --component local/claimed-gw-simple-grid-wrapper-example:0.1 --log_level "INFO" --sgw_source_folder /opt/app-root/src/data/simple_grid_wrapper_source --sgw_target_folder /opt/app-root/src/data/simple_grid_wrapper_target

# you can also store the results in the source folder
claimed --component local/claimed-gw-simple-grid-wrapper-example:0.1 --log_level "INFO" --sgw_source_folder /opt/app-root/src/data/simple_grid_wrapper_source_and_target --sgw_target_folder /opt/app-root/src/data/simple_grid_wrapper_source_and_target
```

### 5.5 Folder Grid Wrapper
It's exactly like the simple grid wrapper but here you lock folder instead of files.
Here are the commands, given you are in the examples/folder_grid_wrapper_example folder of this repository:
```
c3_create_gridwrapper folder_grid_wrapper_example.py -b folder_grid_wrapper
export CLAIMED_DATA_PATH=/path/to/your/c3/examples
claimed --component local/claimed-gw-folder-grid-wrapper-example:0.1 --log_level "INFO" --sgw_source_folder /opt/app-root/src/data/folder_grid_wrapper_source --sgw_target_folder /opt/app-root/src/data/folder_grid_wrapper_target
```
CLAIMED_DATA_PATH specifies the root directory that contains both the source and target folders used by the folder grid wrapper.
For example, if 
```
CLAIMED_DATA_PATH=/c3/examples/folder_grid_wrapper_example
```
then the directory structure should look like this:
```
/c3/examples/folder_grid_wrapper_example/
├── folder_grid_wrapper_source/
├── folder_grid_wrapper_target/
```
