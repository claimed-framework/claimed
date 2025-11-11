
import logging
import os
import re
from c3.parser import ContentParser


class Pythonscript:
    def __init__(self, path):

        self.path = path
        with open(path, 'r') as f:
            self.script = f.read()

        self.name = os.path.basename(path)[:-3].replace('_', '-').lower()
        if '"""' not in self.script:
            logging.warning('Please provide a description of the operator in the first doc string.')
            self.description = self.name
        else:
            self.description = self.script.split('"""')[1].strip()
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
                    if re.search(r'=\s*int\(\s*os', line):
                        type = 'Integer'
                        default = default.strip('\"\'')
                    elif re.search(r'=\s*float\(\s*os', line):
                        type = 'Float'
                        default = default.strip('\"\'')
                    elif re.search(r'=\s*bool\(\s*os', line):
                        type = 'Boolean'
                        default = default.strip('\"\'')
                    else:
                        type = 'String'
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
        # TODO: Does not check for description code
        return_value = {name: {
            'description': f'Output path for {name}',
            'type': 'String',
        } for name in output_names}
        return return_value

    def get_requirements(self):
        requirements = []
        # Add dnf install
        for line in self.script.split('\n'):
            if re.search(r'[\s#]*dnf\s*.[^#]*', line):
                if '-y' not in line:
                    # Adding default repo
                    line += ' -y'
                requirements.append(line.replace('#', '').strip())

        # Add pip install
        pattern = r"^[# !]*(pip[ ]*install)[ ]*(.[^#]*)"
        for line in self.script.split('\n'):
            result = re.findall(pattern, line)
            if len(result) == 1:
                requirements.append((result[0][0] + ' ' + result[0][1].strip()))
        return requirements

    def get_name(self):
        return self.name

    def get_description(self):
        return self.description

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs
