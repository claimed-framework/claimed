"""
This is the operator description.
"""

# pip install numpy

#!pip install pandas

# dnf update

import os
import numpy as np

# A comment one line above os.getenv is the description of this variable.
input_path = os.environ.get('input_path', '')  # ('not this')

# type casting to int(), float(), or bool()
batch_size = int(os.environ.get('batch_size', 16))  # (not this)

# Commas in the previous comment are deleted because the yaml file requires descriptions without commas.
debug = bool(os.getenv('debug', False))

output_path = os.getenv('output_path', 'default_value')


def main(*args):
    """
    The compiler only includes the first doc string.This text should not be included.
    """
    _ = np.random.randn(5)

    os.environ['test_output'] = 'test'

    print(args)


def process(batch, *args):
    # process function for grid wrapper
    print('Execute batch:', batch)
    main(batch, *args, input_path, batch_size, debug, output_path)


if __name__ == '__main__':
    main(input_path, batch_size, debug, output_path)
