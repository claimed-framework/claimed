import argparse
import os
import sys
import logging
import subprocess
import re
from c3.create_operator import create_cwl_component
from c3.pythonscript import Pythonscript
from c3.templates import component_setup_code_wo_logging, python_component_setup_code

def create_containerless_operator(
        file_path,
        version,
        skip_logging = False
    ):

    if version is None:
        version = 'latest'

    logging.debug(f'Called create_containerless_operator {version} with {file_path}')

    filename, file_extension = os.path.splitext(file_path)

    if file_extension != '.py':
        raise NotImplementedError('Containerless operators currenly only support python scripts')
    
    all_pip_packages_found = ''
    with open(file_path, 'r') as file:
        for line in file:
            if re.search('pip ', line):
                pip_packages = re.sub('[#, ,!]*pip[ ]*install[ ]*', '', line)
                logging.debug(f'PIP packages found: {pip_packages}')
                all_pip_packages_found += (f' {pip_packages}')
    logging.info(f'all PIP packages found: {all_pip_packages_found}')


    # prepend init code to script
    target_code = 'runnable.py'

    if os.path.exists(target_code):
        os.remove(target_code)

    with open(file_path, 'r') as f:
        script = f.read()
    if skip_logging:
        script = component_setup_code_wo_logging + script
    else:
        script = python_component_setup_code + script
    with open(target_code, 'w') as f:
        f.write(script)

    subprocess.run(';'.join(['rm -Rf claimedenv','python -m venv claimedenv', 
                                        'source ./claimedenv/bin/activate', 
                                        f'pip install {all_pip_packages_found.strip()}',
                                        'pip list',
                                        f'zip -r claimed-{filename}:{version}.zip {target_code} claimedenv',
                                        'rm -Rf claimedenv',
                                        f'rm {target_code}']), shell=True)
    script_data = Pythonscript(file_path)
    inputs = script_data.get_inputs()
    outputs = script_data.get_outputs()

    create_cwl_component(filename, "containerless", version, file_path, inputs, outputs)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE_PATH', type=str,
                        help='Path to python script or notebook')
    parser.add_argument('ADDITIONAL_FILES', type=str, nargs='*', default=None,
                        help='Paths to additional files to include in the container image')
    parser.add_argument('-v', '--version', type=str, default=None,
                        help='Container image version. Auto-increases the version number if not provided (default 0.1)')
    parser.add_argument('-l', '--log_level', type=str, default='INFO')
    args = parser.parse_args()

    # Init logging
    root = logging.getLogger()
    root.setLevel(args.log_level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(args.log_level)
    root.addHandler(handler)

    create_containerless_operator(
        file_path=args.FILE_PATH,
        version=args.version,
    )

if __name__ == '__main__':
    main()
