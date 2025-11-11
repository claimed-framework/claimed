# TODO: Rename the file to the desired operator name.
"""
TODO: Update the description of the operator in the first doc string.
This is the operator description.
The file name becomes the operator name.
"""

# TODO: Update the required pip packages.
# pip install numpy

import os
import logging
import numpy as np

# TODO: Add the operator interface.
# A comment one line above os.getenv is the description of this variable.
input_path = os.getenv('input_path')

# If you specify a default value, this parameter gets marked as optional
with_default = os.getenv('with_default', 'default_value')

# You can cast to a specific type with int(), float(), or bool() - this type information propagates down to the execution engines (e.g., Kubeflow)
num_values = int(os.getenv('num_values', 5))

# Output paths are starting with "output_".
output_path = os.getenv('output_path', None)


# You can call a function from an additional file (must be in the same directory) or add your code here.
def main(num_values, *args, **kwargs):
    # TODO: Add your code.
    random_values = np.random.rand(num_values)
    # C3 adds setup code to your script which initalize the logging.
    # You can just use logging.debug(), logging.info(), logging.warning() in your code.
    logging.info(f'Random values: {random_values}')


# It is recommended to use a main block to avoid unexpected code execution.
if __name__ == '__main__':
    main(num_values)


# TODO: Add a grid process if you want to parallelize your code.
def grid_process(batch_id, input_path, with_default, num_values, output_path):
    """
    A process for the c3 grid wrapper. The process gets the batch name as the first positional argument,
    followed by all interface variables. This is only possible if the code can be processed in parallel,
    e.g., by splitting up input files.
    """

    # You might need to update the variables based on the batch
    input_path += batch_id + '*.json'
    output_path += batch_id + '_data.csv'

    # Execute the processing with adjusted variables
    main(num_values, input_path, output_path)

    # optionally return a string or list with output files
    return output_path