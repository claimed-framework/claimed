
import os
from string import Template
from pathlib import Path

# template file names
PYTHON_COMPONENT_SETUP_CODE = 'component_setup_code.py'
R_COMPONENT_SETUP_CODE = 'component_setup_code.R'
PYTHON_COMPONENT_SETUP_CODE_WO_LOGGING = 'component_setup_code_wo_logging.py'
PYTHON_DOCKERFILE_FILE = 'python_dockerfile_template'
R_DOCKERFILE_FILE = 'R_dockerfile_template'
KFP_COMPONENT_FILE = 'kfp_component_template.yaml'
KUBERNETES_JOB_FILE = 'kubernetes_job_template.job.yaml'
CWL_COMPONENT_FILE = 'cwl_component_template.cwl'
GRID_WRAPPER_FILE = 'grid_wrapper_template.py'
COS_GRID_WRAPPER_FILE = 'cos_grid_wrapper_template.py'
LEGACY_COS_GRID_WRAPPER_FILE = 'legacy_cos_grid_wrapper_template.py'
S3KV_GRID_WRAPPER_FILE = 's3kv_grid_wrapper_template.py'
SIMPLE_GRID_WRAPPER_FILE = 'simple_grid_wrapper_template.py'
FOLDER_GRID_WRAPPER_FILE = 'folder_grid_wrapper_template.py'

# load templates
template_path = Path(os.path.dirname(__file__))

with open(template_path / PYTHON_COMPONENT_SETUP_CODE, 'r') as f:
    python_component_setup_code = f.read()

with open(template_path / R_COMPONENT_SETUP_CODE, 'r') as f:
    r_component_setup_code = f.read()

with open(template_path / PYTHON_COMPONENT_SETUP_CODE_WO_LOGGING, 'r') as f:
    component_setup_code_wo_logging = f.read()

with open(template_path / PYTHON_DOCKERFILE_FILE, 'r') as f:
    python_dockerfile_template = Template(f.read())

with open(template_path / R_DOCKERFILE_FILE, 'r') as f:
    r_dockerfile_template = Template(f.read())

with open(template_path / KFP_COMPONENT_FILE, 'r') as f:
    kfp_component_template = Template(f.read())

with open(template_path / KUBERNETES_JOB_FILE, 'r') as f:
    kubernetes_job_template = Template(f.read())

with open(template_path / CWL_COMPONENT_FILE, 'r') as f:
    cwl_component_template = Template(f.read())

with open(template_path / GRID_WRAPPER_FILE, 'r') as f:
    grid_wrapper_template = Template(f.read())

with open(template_path / COS_GRID_WRAPPER_FILE, 'r') as f:
    cos_grid_wrapper_template = Template(f.read())

with open(template_path / LEGACY_COS_GRID_WRAPPER_FILE, 'r') as f:
    legacy_cos_grid_wrapper_template = Template(f.read())

with open(template_path / S3KV_GRID_WRAPPER_FILE, 'r') as f:
    s3kv_grid_wrapper_template = Template(f.read())

with open(template_path / SIMPLE_GRID_WRAPPER_FILE, 'r') as f:
    simple_grid_wrapper_template = Template(f.read())

with open(template_path / FOLDER_GRID_WRAPPER_FILE, 'r') as f:
    folder_grid_wrapper_template = Template(f.read())
    