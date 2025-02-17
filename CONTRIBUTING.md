<!--
{% comment %}
Copyright 2018-2024 IBM

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

# Contributing

Welcome to TerraTorch-Iterate! If you are interested in contributing to the [TerraTorch-Iterate code repo](README.md)
then checkout the [Contributing page](CONTRIBUTING.md) and 
the [Code of Conduct](CODE_OF_CONDUCT.md)). 


### Getting Started

It's encouraged that you look under the [Issues]([https://github.ibm.com/GeoFM-Finetuning/benchmark/issues)) tab for contribution opportunites.

### Running tests

Prerequisite:
-  set environment variables:
   - `SEGMENTATION_V1` variable describes the path to segmetation_v1.0 dataset, which must be available locally (please see this example on how to load this dataset https://github.com/ServiceNow/geo-bench/blob/main/geobench/example_load_datasets.py)
   -  `BACKBONE_PRETRAINED_FILE` full path to the pretrained file
   -  `OUTPUT_DIR` directory that contains the results (e.g., mlflow run outputs)
-  a python env has been created (see "Package installation" section in [README](README.md)) and activated

Steps:
1. Go to benchmark dir
2. Run the script: `./run_tests.sh`