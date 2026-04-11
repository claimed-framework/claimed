import os
import re
import sys
import logging

# get parameters from args
parameters = list(filter(
                lambda s: s.find('=') > -1 and bool(re.match(r'[A-Za-z0-9_]*=[.\/A-Za-z0-9]*', s)),
                sys.argv
            ))

# set parameters to env variables
for parameter in parameters:
    variable = parameter.split('=')[0]
    value = parameter.split('=', 1)[-1]
    logging.debug(f'Parameter: {variable} = "{value}"')
    os.environ[variable] = value
