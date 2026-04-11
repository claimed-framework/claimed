
import os
import sys
import logging
import shutil
import argparse
import subprocess
import glob
import re
import json
from pathlib import Path
from string import Template
from typing import Optional
from c3.pythonscript import Pythonscript
from c3.notebook import Notebook
from c3.rscript import Rscript
from c3.utils import convert_notebook, get_image_version
from c3.templates import (python_component_setup_code, component_setup_code_wo_logging, r_component_setup_code,
                          python_dockerfile_template, r_dockerfile_template,
                          kfp_component_template, kubernetes_job_template, cwl_component_template)

CLAIMED_VERSION = 'V0.1'


def create_dockerfile(dockerfile_template, dockerfile, requirements, target_code, target_dir, additional_files,
                      working_dir, command, image_version):
    # Check for requirements file
    for i in range(len(requirements)):
        if '-r ' in requirements[i]:
            r_file_search = re.search('-r ~?\/?([^\s]*\.txt)', requirements[i])
            if len(r_file_search.groups()):
                # Get file from regex
                requirements_file = r_file_search.groups()[0]
                if requirements_file not in additional_files and os.path.isfile(requirements_file):
                    # Add missing requirements text file to additional files
                    additional_files.append(r_file_search.groups()[0])
            if '/' not in requirements[i]:
                # Add missing home directory to the command `pip install -r ~/requirements.txt`
                requirements[i] = requirements[i].replace('-r ', '-r ~/')

    requirements_docker = list(map(lambda s: 'RUN ' + s, requirements))
    requirements_docker = '\n'.join(requirements_docker)
    additional_files_docker = list(map(lambda s: f"ADD {s} {working_dir}{s}", additional_files))
    additional_files_docker = '\n'.join(additional_files_docker)

    # Select base image
    if 'python' in command:
        base_image = f"registry.access.redhat.com/ubi8/python-{image_version.strip('python').replace('.', '')}"
    elif command == 'Rscript':
        if 'python' in image_version:
            # Using default R version
            image_version = 'R4.3.2'
        base_image = f"r-base:{image_version.strip('Rr:')}"
    else:
        raise ValueError(f'Unrecognized command {command}')
    logging.info(f'Using base image {base_image}')

    docker_file = dockerfile_template.substitute(
        base_image=base_image,
        requirements_docker=requirements_docker,
        target_code=target_code,
        target_dir=target_dir,
        additional_files_docker=additional_files_docker,
        working_dir=working_dir,
        command=os.path.basename(command),
    )

    logging.info('Create Dockerfile')
    with open(dockerfile, "w") as text_file:
        text_file.write(docker_file)
    logging.debug(f'{dockerfile}:\n' + docker_file)


def create_kfp_component(name, description, repository, version, command, target_code, target_dir, file_path, inputs, outputs):

    inputs_list = str()
    for input, options in inputs.items():
        inputs_list += f'- {{name: {input}, type: {options["type"]}, description: "{options["description"]}"'
        if options['default'] is not None:
            if not options["default"].startswith('"'):
                options["default"] = f'"{options["default"]}"'
            inputs_list += f', default: {options["default"]}'
        inputs_list += '}\n'

    outputs_list = str()
    for output, options in outputs.items():
        outputs_list += f'- {{name: {output}, type: String, description: "{options["description"]}"}}\n'

    parameter_list = str()
    for index, key in enumerate(list(inputs.keys()) + list(outputs.keys())):
        parameter_list += f'{key}="${{{index}}}" '

    parameter_values = str()
    for input_key in inputs.keys():
        parameter_values += f"        - {{inputValue: {input_key}}}\n"
    for output_key in outputs.keys():
        parameter_values += f"        - {{outputPath: {output_key}}}\n"

    yaml = kfp_component_template.substitute(
        name=name,
        description=description,
        repository=repository,
        version=version,
        inputs=inputs_list,
        outputs=outputs_list,
        command=os.path.basename(command),
        target_dir=target_dir,
        target_code=target_code,
        parameter_list=parameter_list,
        parameter_values=parameter_values,
    )

    logging.debug('KubeFlow component yaml:\n' + yaml)
    target_yaml_path = str(Path(file_path).with_suffix('.yaml'))

    logging.info(f'Write KubeFlow component yaml to {target_yaml_path}')
    with open(target_yaml_path, "w") as text_file:
        text_file.write(yaml)


def create_kubernetes_job(name, repository, version, target_code, target_dir, command, working_dir, file_path, inputs):
    # get environment entries
    env_entries = str()
    for key in list(inputs.keys()):
        env_entries += f"        - name: {key}\n          value: value_of_{key}\n"
    env_entries = env_entries.rstrip()

    job_yaml = kubernetes_job_template.substitute(
        name=name,
        repository=repository,
        version=version,
        target_code=target_code,
        target_dir=target_dir,
        env_entries=env_entries,
        command=command,
        working_dir=working_dir,
    )

    logging.debug('Kubernetes job yaml:\n' + job_yaml)
    target_job_yaml_path = str(Path(file_path).with_suffix('.job.yaml'))

    logging.info(f'Write kubernetes job yaml to {target_job_yaml_path}')
    with open(target_job_yaml_path, "w") as text_file:
        text_file.write(job_yaml)


def create_cwl_component(name, repository, version, file_path, inputs, outputs):
    type_dict = {'String': 'string', 'Integer': 'int', 'Float': 'float', 'Boolean': 'bool'}
    # get environment entries
    i = 1
    input_envs = str()
    for input, options in inputs.items():
        i += 1
        # Convert string default value to CWL types
        default_value = options['default'] if options['type'] == 'String' and options['default'] != '"None"' \
            else options['default'].strip('"\'')
        input_envs += (f"  {input}:\n    type: {type_dict[options['type']]}\n    default: {default_value}\n    "
                       f"inputBinding:\n      position: {i}\n      prefix: --{input}\n")

    if len(outputs) == 0:
        output_envs = '[]'
    else:
        output_envs = '\n'
    for output, options in outputs.items():
        i += 1
        output_envs += (f"  {output}:\n    type: string\n    "
                        f"inputBinding:\n      position: {i}\n      prefix: --{output}\n")

    cwl = cwl_component_template.substitute(
        name=name,
        repository=repository,
        version=version,
        inputs=input_envs,
        outputs=output_envs,
    )

    logging.debug('CWL component:\n' + cwl)
    target_cwl_path = str(Path(file_path).with_suffix('.cwl'))

    logging.info(f'Write cwl component to {target_cwl_path}')
    with open(target_cwl_path, "w") as text_file:
        text_file.write(cwl)


def check_existing_files(file_path, rename_files, overwrite_files):
    if rename_files is None and overwrite_files:
        # Overwrite potential files
        return

    target_job_yaml_path = Path(file_path).with_suffix('.job.yaml')

    # Check for existing job yaml
    if target_job_yaml_path.is_file():
        if rename_files is None:
            # Ask user
            rename_files = input(f'\nFound a existing Kubernetes job file at {target_job_yaml_path}.\n'                      
                                 f'ENTER to overwrite the file, write Y to rename the file to '
                                 f'modified_{target_job_yaml_path.name}, or provide a custom name:\n')
        if rename_files.strip() == '':
            # Overwrite file
            return
        elif rename_files.lower() == 'y':
            # Default file name
            new_file_name = 'modified_' + Path(file_path).name
        else:
            # Rename to custom name
            new_file_name = rename_files

        modified_path = (target_job_yaml_path.parent / new_file_name).with_suffix('.job.yaml')
        # Check if modified path exists and potentially overwrite
        if modified_path.exists():
            if overwrite_files:
                logging.info(f'Overwriting modified path {modified_path}.')
            else:
                overwrite = input(f'Modified path {modified_path} already exists. ENTER to overwrite the file.')
                if overwrite != '':
                    logging.error(f'Abort creating operator. Please rename file manually and rerun the script.')
                    raise FileExistsError

        os.rename(str(target_job_yaml_path), str(modified_path))
        logging.info(f'Renamed Kubernetes job file to {modified_path}')
    # TODO: Should we check other files too? Currently assuming no modification for yaml and cwl.


def print_claimed_command(name, repository, version, inputs):
    claimed_command = f"claimed --component {repository}/claimed-{name}:{version}"
    for input, options in inputs.items():
        claimed_command += f" --{input} {options['default']}"
    logging.info(f'Run operators locally with claimed-cli:\n{claimed_command}')


def remove_temporary_files(file_path, target_code):
    logging.info(f'Remove local files')
    # remove temporary files
    if file_path != target_code:
        os.remove(target_code)
    if os.path.isfile('Dockerfile'):
        os.remove('Dockerfile')


def create_operator(file_path: str,
                    repository: str,
                    version: str,
                    custom_dockerfile_template: Optional[Template],
                    additional_files: str = None,
                    log_level='INFO',
                    local_mode=False,
                    no_cache=False,
                    rename_files=None,
                    overwrite_files=False,
                    skip_logging=False,
                    keep_generated_files=False,
                    platform='linux/amd64',
                    dockerfile='Dockerfile.generated',
                    image_version='python3.12',
                    skip_docker_build=False,
                    ):
    logging.info('Parameters: ')
    logging.info('file_path: ' + file_path)
    logging.info('repository: ' + str(repository))
    logging.info('version: ' + str(version))
    logging.info('additional_files: ' + '; '.join(additional_files))

    if file_path.endswith('.py'):
        # use temp file for processing
        target_code = 'claimed_' + os.path.basename(file_path)
        # Copy file to current working directory
        shutil.copy(file_path, target_code)
        # Add code for logging and cli parameters to the beginning of the script
        with open(target_code, 'r') as f:
            script = f.read()
        if skip_logging:
            script = component_setup_code_wo_logging + script
        else:
            script = python_component_setup_code + script
        with open(target_code, 'w') as f:
            f.write(script)
        # getting parameter from the script
        script_data = Pythonscript(target_code)
        dockerfile_template = custom_dockerfile_template or python_dockerfile_template
        command = '/opt/app-root/bin/python'
        working_dir = '/opt/app-root/src/'

    elif file_path.endswith('.ipynb'):
        # use temp file for processing
        target_code = 'claimed_' + os.path.basename(file_path)
        # Copy file to current working directory
        shutil.copy(file_path, target_code)
        with open(target_code, 'r') as json_file:
            notebook = json.load(json_file)
        # Add code for logging and cli parameters to the beginning of the notebook
        notebook['cells'].insert(0, {
            'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': component_setup_code_wo_logging if skip_logging else python_component_setup_code})
        with open(target_code, 'w') as json_file:
             json.dump(notebook, json_file)
        # getting parameter from the script
        script_data = Notebook(target_code)
        dockerfile_template = custom_dockerfile_template or python_dockerfile_template
        command = '/opt/app-root/bin/ipython'
        working_dir = '/opt/app-root/src/'

    elif file_path.lower().endswith('.r'):
        # use temp file for processing
        target_code = 'claimed_' + os.path.basename(file_path)
        # Copy file to current working directory
        shutil.copy(file_path, target_code)
        # Add code for logging and cli parameters to the beginning of the script
        with open(target_code, 'r') as f:
            script = f.read()
        script = r_component_setup_code + script
        with open(target_code, 'w') as f:
            f.write(script)
        # getting parameter from the script
        script_data = Rscript(target_code)
        dockerfile_template = custom_dockerfile_template or r_dockerfile_template
        command = 'Rscript'
        working_dir = '/home/docker/'
    else:
        raise NotImplementedError('Please provide a file_path to a jupyter notebook, python script, or R script.')

    name = script_data.get_name()
    # convert description into a string with a single line
    description = ('"' + script_data.get_description().replace('\n', ' ').replace('"', '\'') +
                   ' – CLAIMED ' + CLAIMED_VERSION + '"')
    inputs = script_data.get_inputs()
    outputs = script_data.get_outputs()
    requirements = script_data.get_requirements()
    # Strip 'claimed-' from name of copied temp file
    if name.startswith('claimed-'):
        name = name[8:]
    target_dir = os.path.dirname(file_path)
    # Check that the main file is within the cwd
    if '../' in target_dir:
        raise PermissionError(f"Forbidden path outside the docker build context: {target_dir}. "
                              f"Change the current working directory to include the file.")
    elif target_dir != '':
        target_dir += '/'

    logging.info('Operator name: ' + name)
    logging.info('Description: ' + description)
    logging.info('Inputs:\n' + ('\n'.join([f'{k}: {v}' for k, v in inputs.items()])))
    logging.info('Outputs:\n' + ('\n'.join([f'{k}: {v}' for k, v in outputs.items()])))
    logging.info('Requirements: ' + '; '.join(requirements))
    logging.debug(f'Target code: {target_code}')
    logging.debug(f'Target directory: {target_dir}')

    # Load all additional files
    logging.debug('Looking for additional files:')
    additional_files_found = []
    for file_pattern in additional_files:
        if '../' in file_pattern:
            # Check that additional file are within the cwd
            raise PermissionError(f"Forbidden path outside the docker build context: {file_pattern}. "
                                  f"Change the current working directory to include all additional files.")
        # Include files based on wildcards
        files_found = glob.glob(file_pattern)
        if len(files_found) == 0:
            raise FileNotFoundError(f'No additional files for path {file_pattern}.')
        additional_files_found.extend(files_found)
        logging.debug(f'Searched for "{file_pattern}". Found {", ".join(files_found)}')
    logging.info(f'Found {len(additional_files_found)} additional files and directories\n'
                 f'{", ".join(additional_files_found)}')

    create_dockerfile(dockerfile_template, dockerfile, requirements, target_code, target_dir, additional_files_found,
                      working_dir, command, image_version)

    if version is None:
        # auto increase version based on registered images
        version = get_image_version(repository, name)

    if repository is None:
        if not local_mode:
            logging.warning('No repository provided. The container image is only saved locally. Add `-r <repository>` '
                            'to push the image to a container registry or run `--local_mode` to suppress this warning.')
        local_mode = True
        repository = 'local'

    if not skip_docker_build:                   
        if subprocess.run('docker buildx', shell=True, stdout=subprocess.PIPE).returncode == 0:
            # Using docker buildx
            logging.debug('Using docker buildx')
            build_command = f'docker buildx build -f {dockerfile}'
        else:
            logging.debug('Using docker build. Consider installing docker-buildx.')
            build_command = f'docker build -f {dockerfile}'
    
        logging.info(f'Building container image claimed-{name}:{version}')
        try:
            # Run docker build
            subprocess.run(
                f"{build_command} --platform {platform} -t claimed-{name}:{version} . {'--no-cache' if no_cache else ''}",
                stdout=None if log_level == 'DEBUG' else subprocess.PIPE, check=True, shell=True
            )
            if repository is not None:
                # Run docker tag
                logging.debug(f'Tagging images with "latest" and "{version}"')
                subprocess.run(
                    f"docker tag claimed-{name}:{version} {repository}/claimed-{name}:{version}",
                    stdout=None if log_level == 'DEBUG' else subprocess.PIPE, check=True, shell=True,
                )
                subprocess.run(
                    f"docker tag claimed-{name}:{version} {repository}/claimed-{name}:latest",
                    stdout=None if log_level == 'DEBUG' else subprocess.PIPE, check=True, shell=True,
                )
        except Exception as err:
            logging.error('Docker build failed. Consider running C3 with `--log_level DEBUG` to see the docker build logs.')
            if not keep_generated_files:
                remove_temporary_files(file_path, target_code)
            raise err
        logging.info(f'Successfully built image claimed-{name}:{version}')
    
        if local_mode:
            logging.info(f'No repository provided, skip docker push.')
        else:
            logging.info(f'Pushing images to registry {repository}')
            try:
                # Run docker push
                subprocess.run(
                    f"docker push {repository}/claimed-{name}:latest",
                    stdout=None if log_level == 'DEBUG' else subprocess.PIPE, check=True, shell=True,
                )
                subprocess.run(
                    f"docker push {repository}/claimed-{name}:{version}",
                    stdout=None if log_level == 'DEBUG' else subprocess.PIPE, check=True, shell=True,
                )
                logging.info('Successfully pushed image to registry')
            except Exception as err:
                logging.error(f'Could not push images to namespace {repository}. '
                              f'Please check if docker is logged in or select a namespace with access.')
                if not keep_generated_files:
                    remove_temporary_files(file_path, target_code)
                raise err

    # Check for existing files and optionally modify them before overwriting
    try:
        check_existing_files(file_path, rename_files, overwrite_files)
    except Exception as err:
        if not keep_generated_files:
            remove_temporary_files(file_path, target_code)
        raise err

    # Create application scripts
    create_kfp_component(name, description, repository, version, command, target_code, target_dir, file_path, inputs,
                         outputs)

    create_kubernetes_job(name, repository, version, target_code, target_dir, command, working_dir, file_path, inputs)

    create_cwl_component(name, repository, version, file_path, inputs, outputs)

    print_claimed_command(name, repository, version, inputs)

    # Remove temp files
    if not keep_generated_files:
        remove_temporary_files(file_path, target_code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE_PATH', type=str,
                        help='Path to python script or notebook')
    parser.add_argument('ADDITIONAL_FILES', type=str, nargs='*',
                        help='Paths to additional files to include in the container image')
    parser.add_argument('-r', '--repository', type=str, default=None,
                        help='Container registry address, e.g. docker.io/<username>')
    parser.add_argument('-v', '--version', type=str, default=None,
                        help='Container image version. Auto-increases the version number if not provided (default 0.1)')
    parser.add_argument('--rename', type=str, nargs='?', default=None, const='',
                        help='Rename existing yaml files (argument without value leads to modified_{file name})')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing yaml files')
    parser.add_argument('-l', '--log_level', type=str, default='INFO')
    parser.add_argument('--dockerfile_template_path', type=str, default='',
                        help='Path to custom dockerfile template')
    parser.add_argument('--dockerfile', type=str, default='Dockerfile.generated',
                        help='Name or path of the generated dockerfile.')
    parser.add_argument('--local_mode', action='store_true',
                        help='Continue processing after docker errors.')
    parser.add_argument('--no-cache', action='store_true', help='Not using cache for docker build.')
    parser.add_argument('--skip-logging', action='store_true',
                        help='Exclude logging code from component setup code')
    parser.add_argument('--keep-generated-files', action='store_true',
                        help='Do not delete temporary generated files.')
    parser.add_argument('--platform', type=str, default='linux/amd64',
                        help='Select image platform, default is linux/amd64. Alternativly, select linux/arm64".')
    parser.add_argument('--image_version', type=str, default='python3.12',
                        help='Select python or R version (defaults to python3.12).')
    parser.add_argument('--skip-docker-build', action='store_true',
                        help='Enable skipping docker build (default: False).')

    args = parser.parse_args()

    # Init logging
    root = logging.getLogger()
    root.setLevel(args.log_level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(args.log_level)
    root.addHandler(handler)

    # Update dockerfile template if specified
    if args.dockerfile_template_path != '':
        logging.info(f'Uses custom dockerfile template from {args.dockerfile_template_path}')
        with open(args.dockerfile_template_path, 'r') as f:
            custom_dockerfile_template = Template(f.read())
    else:
        custom_dockerfile_template = None

    create_operator(
        file_path=args.FILE_PATH,
        repository=args.repository,
        version=args.version,
        custom_dockerfile_template=custom_dockerfile_template,
        additional_files=args.ADDITIONAL_FILES,
        log_level=args.log_level,
        local_mode=args.local_mode,
        no_cache=args.no_cache,
        overwrite_files=args.overwrite,
        rename_files=args.rename,
        skip_logging=args.skip_logging,
        keep_generated_files=args.keep_generated_files,
        platform=args.platform,
        dockerfile=args.dockerfile,
        image_version=args.image_version,
        skip_docker_build=args.skip_docker_build,
    )


if __name__ == '__main__':
    main()
