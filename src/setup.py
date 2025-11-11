from setuptools import setup, find_packages

setup(
    name='c3',
    packages=find_packages(),
    install_requires=[
        'ipython',
        'nbconvert',
    ],
)
