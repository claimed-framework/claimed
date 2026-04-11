
import logging
import os
import re
from c3.parser import ContentParser


class Rscript:
    def __init__(self, path):

        self.path = path
        with open(path, 'r') as f:
            self.script = f.read()

        self.name = os.path.basename(path)[:-2].replace('_', '-').lower()
        # TODO: Currently does not support a description
        self.description = self.name
        self.inputs = self._get_input_vars()
        self.outputs = self._get_output_vars()

    def _get_input_vars(self):
        cp = ContentParser()
        env_names = cp.parse(self.path)['inputs']
        return_value = dict()
        for env_name, default in env_names.items():
            comment_line = str()
            for line in self.script.split('\n'):
                if re.search("[\"']" + env_name + "[\"']", line):
                    # Check the description for current variable
                    if not comment_line.strip().startswith('#'):
                        # previous line was no description, reset comment_line.
                        comment_line = ''
                    if comment_line == '':
                        logging.debug(f'Interface: No description for variable {env_name} provided.')
                    if re.search(r'=\s*as.numeric\(\s*os', line):
                        type = 'Float'  # double in R
                    elif re.search(r'=\s*bool\(\s*os', line):
                        type = 'Boolean'  # logical in R
                    else:
                        type = 'String'  # character in R

                    return_value[env_name] = {
                        'description': comment_line.replace('#', '').replace("\"", "\'").strip(),
                        'type': type,
                        'default': default
                    }
                    break
                comment_line = line
        return return_value

    def _get_output_vars(self):
        cp = ContentParser()
        output_names = cp.parse(self.path)['outputs']
        # TODO: Does not check for description
        return_value = {name: {'description': 'output path'} for name in output_names}
        return return_value

    def get_requirements(self):
        requirements = []
        # Add apt install commands
        for line in self.script.split('\n'):
            if re.search(r'[\s#]*apt\s*[A-Za-z0-9_-]*', line):
                if '-y' not in line:
                    # Adding default repo
                    line += ' -y'
                requirements.append(line.replace('#', '').strip())

        # Add Rscript install.packages commands
        for line in self.script.split('\n'):
            if re.search(r'[\s#]*install\.packages\(.*\)', line):
                if 'http://' not in line:
                    # Adding default repo
                    line = line.rstrip(') ') + ", repos='http://cran.us.r-project.org')"
                command = f"Rscript -e \"{line.replace('#', '').strip()}\""
                requirements.append(command)
        return requirements

    def get_name(self):
        return self.name

    def get_description(self):
        return self.description

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs
