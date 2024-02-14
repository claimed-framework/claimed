# The workbench close to the library

This is just an example folder to tell you that not all your code needs to be part of the package.

For example there could be `experiments/` with training scripts or `notebooks/` with jupyter notebooks where you try stuff.

This stuff will still be under version control and can conveniently import from the package.

Files here will not be installed, but could be configured to be via `package_data` in `setup.py`.
