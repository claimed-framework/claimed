import logging
import os
import argparse
import sys
from string import Template
from c3.pythonscript import Pythonscript
from c3.utils import convert_notebook
from c3.create_operator import create_operator
from c3.templates import component_setup_code_wo_logging
import c3


def wrap_component(component_path,
                   component_description,
                   component_dependencies,
                   component_interface,
                   component_inputs,
                   component_process,
                   backend,
                   ):
    # get component name from path
    component_name = os.path.splitext(os.path.basename(component_path))[0]

    logging.info(f'Using backend: {backend}')

    backends = {
        'local': c3.templates.grid_wrapper_template,
        'cos': c3.templates.cos_grid_wrapper_template,
        'legacy_cos': c3.templates.legacy_cos_grid_wrapper_template,
        's3kv': c3.templates.s3kv_grid_wrapper_template,
        'grid_wrapper': c3.templates.grid_wrapper_template,
        'cos_grid_wrapper': c3.templates.cos_grid_wrapper_template,
        'legacy_cos_grid_wrapper': c3.templates.legacy_cos_grid_wrapper_template,
        's3kv_grid_wrapper': c3.templates.s3kv_grid_wrapper_template,
        'simple_grid_wrapper': c3.templates.simple_grid_wrapper_template,
        'folder_grid_wrapper': c3.templates.folder_grid_wrapper_template,
    }
    gw_template = backends.get(backend)

    logging.debug(f'Using backend template: {gw_template}')

    grid_wrapper_code = gw_template.substitute(
        component_name=component_name,
        component_description=component_description,
        component_dependencies=component_dependencies,
        component_inputs=component_inputs,
        component_interface=component_interface,
        component_process=component_process,
    )

    # Write edited code to file
    grid_wrapper_file = f'gw_{component_name}.py'
    grid_wrapper_file_path = os.path.join(os.path.dirname(component_path), grid_wrapper_file)
    # remove 'component_' from gw path
    grid_wrapper_file_path = grid_wrapper_file_path.replace('component_', '')
    with open(grid_wrapper_file_path, 'w') as f:
        f.write(grid_wrapper_code)

    logging.info(f'Saved wrapped component to {grid_wrapper_file_path}')

    return grid_wrapper_file_path


def get_component_elements(file_path):
    # get required elements from component code
    py = Pythonscript(file_path)
    # convert description into a string with a single line
    description = (py.get_description().replace('\n', ' ').replace('"', '\''))
    inputs = py.get_inputs()
    outputs = py.get_outputs()
    dependencies = py.get_requirements()

    # combine dependencies list
    dependencies = '\n# '.join(dependencies)

    # generate interface code from inputs
    interface = ''
    type_to_func = {'String': '', 'Boolean': 'bool', 'Integer': 'int', 'Float': 'float'}
    for variable, d in inputs.items():
        interface += f"# {d['description']}\n"
        if (d['type'] == 'String' and d['default'] is not None and
            (d['default'] == '' or d['default'][0] not in '\'\"')):
            # Add quotation marks
            d['default'] = "'" + d['default'] + "'"
        interface += f"component_{variable} = {type_to_func[d['type']]}(os.getenv('{variable}', {d['default']}))\n"

    # TODO: Implement output interface
    if len(outputs) > 0:
        logging.warning('Found output paths in the component code which is currently not supported.')

    # generate kwargs for the subprocesses
    process_inputs = ', '.join([f'{i}=component_{i}' for i in inputs.keys()])
    # use log level from grid wrapper
    process_inputs = process_inputs.replace('component_log_level', 'log_level')

    return description, interface, process_inputs, dependencies


# Adding code
def edit_component_code(file_path, component_process):
    file_name = os.path.basename(file_path)
    if file_path.endswith('.ipynb'):
        logging.info('Convert notebook to python script')
        target_file = convert_notebook(file_path)
        file_path = target_file
        file_name = os.path.basename(file_path)
    else:
        # write edited code to different file
        target_file = os.path.join(os.path.dirname(file_path), 'component_' + file_name.replace('-', '_'))

    target_file_name = os.path.basename(target_file)

    with open(file_path, 'r') as f:
        script = f.read()
    assert component_process in script, (f'Did not find the grid process {component_process} in the script. '
                                         f'Please provide the grid process in the arguments `-p <grid_process>`.')
    # Add code for logging and cli parameters to the beginning of the script
    script = component_setup_code_wo_logging + script
    # replace old filename with new file name
    script = script.replace(file_name, target_file_name)
    with open(target_file, 'w') as f:
        f.write(script)

    if '__main__' not in script:
        logging.warning('No __main__ found in component code. Grid wrapper will import functions from component, '
                        'which can lead to unexpected behaviour without using __main__.')

    logging.info('Saved component python script in ' + target_file)

    return target_file


def apply_grid_wrapper(file_path, component_process, backend):
    assert file_path.endswith('.py') or file_path.endswith('.ipynb'), \
        "Please provide a component file path to a python script or notebook."

    file_path = edit_component_code(file_path, component_process)

    description, interface, inputs, dependencies = get_component_elements(file_path)

    component_elements = dict(
        component_path=file_path,
        component_description=description,
        component_dependencies=dependencies,
        component_interface=interface,
        component_inputs=inputs,
        component_process=component_process
    )

    logging.debug('Wrap component with parameters:')
    for component, value in component_elements.items():
        logging.debug(component + ':\n' + str(value) + '\n')

    logging.info('Wrap component')
    grid_wrapper_file_path = wrap_component(backend=backend, **component_elements)
    return grid_wrapper_file_path, file_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE_PATH', type=str,
                        help='Path to python script or notebook')
    parser.add_argument('ADDITIONAL_FILES', type=str, nargs='*',
                        help='List of paths to additional files to include in the container image')
    parser.add_argument('-p', '--component_process', type=str, default='grid_process',
                        help='Name of the component sub process that is executed for each batch.')
    parser.add_argument('-b', '--backend', type=str, default='local',
                        help='Define backend. Default: local. Others: cos, s3kv, legacy_cos (with automatic file download/upload)')
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

    args = parser.parse_args()

    # Init logging
    root = logging.getLogger()
    root.setLevel(args.log_level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(args.log_level)
    root.addHandler(handler)

    grid_wrapper_file_path = component_path = ''
    try:
        grid_wrapper_file_path, component_path = apply_grid_wrapper(
            file_path=args.FILE_PATH,
            component_process=args.component_process,
            backend=args.backend,
        )

        logging.info('Generate CLAIMED operator for grid wrapper')

        # Add component path and init file path to additional_files
        args.ADDITIONAL_FILES.append(component_path)

        # Update dockerfile template if specified
        if args.dockerfile_template_path != '':
            logging.info(f'Uses custom dockerfile template from {args.dockerfile_template_path}')
            with open(args.dockerfile_template_path, 'r') as f:
                custom_dockerfile_template = Template(f.read())
        else:
            custom_dockerfile_template = None

        create_operator(
            file_path=grid_wrapper_file_path,
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
        )
    except Exception as err:
        logging.error('Error while generating CLAIMED grid wrapper. '
                      'Consider using `--log_level DEBUG` and `--keep-generated-files` for debugging.')
        raise err
    finally:
        if not args.keep_generated_files:
            logging.info('Remove local component file and grid wrapper code.')
            if os.path.isfile(grid_wrapper_file_path):
                os.remove(grid_wrapper_file_path)
            if os.path.isfile(component_path):
                os.remove(component_path)


if __name__ == '__main__':
    main()
