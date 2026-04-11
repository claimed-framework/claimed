# default code for each  operator
import os
import sys
import re
import logging

# init logger
root = logging.getLogger()
root.setLevel('INFO')
handler = logging.StreamHandler(sys.stdout)
handler.setLevel('INFO')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
logging.basicConfig(level=logging.CRITICAL)

# get parameters from args
parameters = list(filter(
                lambda s: s.find('=') > -1 and bool(re.match(r'[A-Za-z0-9_]*=[.\/A-Za-z0-9]*', s)),
                sys.argv
            ))

# set parameters to env variables
for parameter in parameters:
    variable = parameter.split('=')[0]
    value = parameter.split('=', 1)[-1]
    logging.info(f'Parameter: {variable} = "{value}"')
    os.environ[variable] = value

# update log level
log_level = os.environ.get('log_level', 'INFO')
if log_level !='INFO':
    logging.info(f'Updating log level to {log_level}')
    root.setLevel(log_level)
    handler.setLevel(log_level)
