import json
import re
import os
import logging
from c3.parser import ContentParser, NotebookReader


class Notebook():
    def __init__(self, path):
        self.path = path
        with open(path) as json_file:
            self.notebook = json.load(json_file)

        self.name = os.path.basename(path)[:-6].replace('_', '-').lower()

        if self.notebook['cells'][1]['cell_type'] == self.notebook['cells'][2]['cell_type'] == 'markdown':
            # backwards compatibility (v0.1 description was included in second cell, merge first two markdown cells)
            logging.info('Merge first two markdown cells for description. '
                         'The file name is used as the operator name, not the first markdown cell.')
            self.description = self.notebook['cells'][1]['source'][0] + '\n' + self.notebook['cells'][2]['source'][0]
        else:
            # Using second cell because first cell was added for setup code
            self.description = self.notebook['cells'][1]['source'][0]

        self.inputs = self._get_input_vars()
        self.outputs = self._get_output_vars()

    def _get_input_vars(self):
        cp = ContentParser()
        env_names = cp.parse(self.path)['inputs']
        return_value = dict()
        notebook_code_lines = list(NotebookReader(self.path).read_next_code_line())
        for env_name, default in env_names.items():
            comment_line = str()
            for line in notebook_code_lines:
                if re.search("[\"']" + env_name + "[\"']", line):
                    if not comment_line.strip().startswith('#'):
                        # previous line was no description, reset comment_line.
                        comment_line = ''
                    if comment_line == '':
                        logging.debug(f'Interface: No description for variable {env_name} provided.')
                    if re.search(r'=\s*int\(\s*os', line):
                        type = 'Integer'
                    elif re.search(r'=\s*float\(\s*os', line):
                        type = 'Float'
                    elif re.search(r'=\s*bool\(\s*os', line):
                        type = 'Boolean'
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
        notebook_code_lines = list(NotebookReader(self.path).read_next_code_line())
        # Add dnf install
        for line in notebook_code_lines:
            if re.search(r'[\s#]*dnf\s*.[^#]*', line):
                if '-y' not in line:
                    # Adding default repo
                    line += ' -y'
                requirements.append(line.replace('#', '').strip())

        # Add pip install
        pattern = r"^[# !]*(pip[ ]*install)[ ]*(.[^#]*)"
        for line in notebook_code_lines:
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
