[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/6718/badge)](https://bestpractices.coreinfrastructure.org/projects/6718)
[![GitHub](https://img.shields.io/badge/issue_tracking-github-blue.svg)](https://github.com/claimed-framework/component-library/issues)



# C3 - the CLAIMED Component Compiler

**TL;DR**
- takes arbitrary assets (Jupyter notebooks, python scripts, R scripts) as input
- automatically creates container images and pushes to container registries
- automatically installs all required dependencies into the container image
- creates KubeFlow Pipeline components (target workflow execution engines are pluggable)
- creates Kubernetes job configs for execution on Kubernetes/Openshift clusters
- can be triggered from CICD pipelines


To learn more on how this library works in practice, please have a look at the following [video](https://www.youtube.com/watch?v=FuV2oG55C5s)

## Related work
[Ploomber](https://github.com/ploomber/ploomber)

[Orchest](https://www.orchest.io/)

## Getting started 

### Install

```sh
pip install claimed
```

### Usage

Just run the following command with your python script or notebook: 
```sh
c3_create_operator "<your-operator-script>.py" --repository "<registry>/<namespace>"
```

Your code needs to follow certain requirements which are explained in [Getting Started](https://github.com/claimed-framework/c3/blob/main/GettingStarted.md). 


## Getting Help

```sh
c3_create_operator --help
```

We welcome your questions, ideas, and feedback. Please create an [issue](https://github.com/claimed-framework/component-library/issues) or a [discussion thread](https://github.com/claimed-framework/component-library/discussions).
Please see [VULNERABILITIES.md](VULNERABILITIES.md) for reporting vulnerabilities.

## Contributing to CLAIMED
Interested in helping make CLAIMED better? We encourage you to take a look at our 
[Contributing](CONTRIBUTING.md) page.

## Credits

CLAIMED is supported by the EU’s Horizon Europe program under Grant Agreement number 101131841 and also received funding from the Swiss State Secretariat for Education, Research and Innovation (SERI) and the UK Research and Innovation (UKRI).

## License
This software is released under Apache License v2.0.
