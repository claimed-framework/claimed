# WLM Plugins

This directory contains reference implementations of **iterate2 WLM plugins** –
executable scripts that submit, wait for, and validate individual Optuna trials
on different workload managers.

## How the plugin system works

When you pass `--wlm-plugin /path/to/plugin` to `iterate2`, it is invoked
**once per trial** with a set of environment variables that describe the work
to be done.  The plugin is responsible for:

1. Submitting the trial to the cluster / WLM.
2. Waiting until the job completes.
3. Ensuring trial stdout is written to `$ITERATE_OUT_FILE` and stderr to
   `$ITERATE_ERR_FILE` (iterate2 extracts metrics from these files).
4. Exiting **0** on success, non-zero on failure.

iterate2 marks the Optuna trial as **FAILED** on a non-zero exit code.

### Environment variables provided by iterate2

| Variable | Description |
|---|---|
| `ITERATE_TRIAL_NUMBER` | Integer trial ID |
| `ITERATE_TRIAL_CMD` | Full shell command (with `cd`, `source venv`, etc.) – use for SSH / HPC WLMs |
| `ITERATE_TRIAL_CONTAINER_CMD` | Bare CLI invocation (no `cd`/`source`) – use for container-based WLMs (Vela/k8s) |
| `ITERATE_OUT_FILE` | Path where **stdout** must be written |
| `ITERATE_ERR_FILE` | Path where **stderr** must be written |
| `ITERATE_WLM_<KEY>` | Every key from the `wlm:` YAML section, uppercased with hyphens→underscores |

### WLM configuration in the HPO YAML

All WLM-specific settings (GPU count, queue, job template path, …) live in the
`wlm:` section of the HPO YAML.  This keeps the launch script clean:

```yaml
# my_hpo.yaml
hpo:
  lr:
    type: float
    low: 1e-5
    high: 1e-2
    log: true

static:
  epochs: 50

wlm:                      # keys forwarded as ITERATE_WLM_* env vars
  gpu-count: 1
  cpu-count: 8
  mem-gb: 32
```

The corresponding launch script only needs:

```bash
iterate2 \
  --script      train.py \
  --wlm-plugin  wlm_plugins/lsf_plugin.sh \
  --hpo-yaml    my_hpo.yaml \
  ...
```

## Provided plugins

| Plugin | WLM | Notes |
|---|---|---|
| [`lsf_plugin.sh`](lsf_plugin.sh) | IBM Spectrum LSF | Uses `bsub -K`; reads `gpu-count`, `cpu-count`, `mem-gb`, `lsf-gpu-config`, `queue` from the `wlm:` section |
| [`vela_plugin.py`](vela_plugin.py) | OpenShift / MLBatch PyTorchJob | Uses `helm template \| oc create`; reads `job-template`, `chart-path`, `namespace`, `cmd-placeholder`, `pod-ready-timeout`, `job-timeout` from the `wlm:` section |

## Writing your own plugin

Any executable (shell script, Python script, compiled binary) works.  Minimal
example that runs the trial locally:

```bash
#!/usr/bin/env bash
# trivial_plugin.sh – run trial locally, redirect output to the log files
bash -c "${ITERATE_TRIAL_CMD}" \
  > "${ITERATE_OUT_FILE}" 2> "${ITERATE_ERR_FILE}"
```

For a SLURM example you can follow the same pattern as `lsf_plugin.sh`,
replacing `bsub` with `srun` / `sbatch`.
