# iterate2 – HPO Launcher

`iterate2` is a generic Optuna-based hyperparameter optimisation (HPO) launcher with a pluggable workload-manager backend. It submits one trial per Optuna suggestion, waits for the job to finish, extracts one or more metrics from the job's log file, and returns them to Optuna.

Key capabilities:

- **Multi-objective optimisation** — extract and optimise several metrics simultaneously (Pareto front)
- **Five HPO parameter types** — `float`, `int`, `categorical`, `flag` (store-true), `group` (bundled arg sets)
- **Dynamic GPU count per trial** — `gpu_num` in the HPO space is passed to the WLM plugin via `ITERATE_WLM_GPU_COUNT`
- **Null-omission** — `null` in a `categorical` choice causes the flag to be completely absent from the command line
- **WLM plugin system** — any executable (bash, Python, …) can be used as a workload-manager backend; reference implementations for LSF and Vela/OpenShift are in `examples/wlm_plugins/`

## Quick start

```sh
iterate2 \
  --script train.py \
  --wlm-plugin examples/wlm_plugins/lsf_plugin.sh \
  --optuna-study-name my_study \
  --optuna-db-path sqlite:///hpo.db \
  --optuna-n-trials 50 \
  --hpo-yaml hpo_space.yaml   # wlm: section sets gpu-count, cpu-count, …
```

For local execution (no cluster) simply omit `--wlm-plugin`:

```sh
iterate2 \
  --script train.py \
  --optuna-study-name my_study \
  --optuna-db-path sqlite:///hpo.db \
  --optuna-n-trials 50 \
  --hpo-yaml hpo_space.yaml
```

## CLI reference

### Execution options

| Option | Default | Description |
|---|---|---|
| `--script` | *(required)* | Training script to execute |
| `--root-dir` | `.` | Working directory; derived from `--script` if omitted |
| `--venv` | *(none)* | Virtual-environment directory to activate. Omit to skip venv activation entirely |
| `--interpreter` | `python` | Python interpreter to invoke |
| `--param-setter` | `None` | Use setter-style argument passing (see [Setter-style arguments](#setter-style-arguments)) |
| `--wlm-plugin` | *(local)* | Path to an executable WLM plugin script. When omitted, trials run locally in the current process |
| `--parallelism` | `1` | Number of trials to run in parallel (see [Parallel execution](#parallel-execution)) |

### Optuna options

| Option | Default | Description |
|---|---|---|
| `--optuna-study-name` | *(required)* | Name of the Optuna study |
| `--optuna-db-path` | *(required)* | Storage URL. `sqlite:///hpo.db` for SQLite, `js:///path/journal.log` for JournalStorage, or any Optuna-supported URL |
| `--optuna-n-trials` | `100` | Number of trials to run |

### HPO search space

Provide either a JSON string or a YAML file that defines the search space under the key `hpo`:

| Option | Description |
|---|---|
| `--hpo-json` | HPO search space as a JSON string |
| `--hpo-yaml` | Path to a YAML file containing the search space (and optionally static args) |

#### Search space YAML format

```yaml
hpo:
  learning_rate:
    type: float
    low: 1e-5
    high: 1e-2
    log: true
  batch_size:
    type: categorical
    choices: [16, 32, 64]
  encoder_depth:
    type: int
    low: 2
    high: 6

static:
  max_epochs: 50
  dataset_path: /data/my_dataset
```

Supported parameter types: `float`, `int`, `categorical`, `flag`, `group`.

#### Parameter types

##### `float`

Suggests a floating-point value between `low` and `high`. Set `log: true` for log-uniform sampling.

```yaml
learning_rate:
  type: float
  low: 1e-5
  high: 1e-2
  log: true
```

Generates: `--learning-rate 0.0003`

##### `int`

Suggests an integer between `low` and `high` (inclusive).

```yaml
encoder_depth:
  type: int
  low: 2
  high: 8
```

Generates: `--encoder-depth 4`

##### `categorical`

Suggests one value from a list of choices. Choices can be strings, numbers, or `null`.

```yaml
batch_size:
  type: categorical
  choices: [16, 32, 64]
```

Generates: `--batch-size 32`

**`null` omits the flag entirely.** Useful for optional flags like `--compile`:

```yaml
compile:
  type: categorical
  choices: ["max-autotune", "default", null]
  # null → --compile is completely absent from the command
```

##### `flag`

Models a `store_true`-style flag that takes no value — its presence or absence is the parameter. `true` adds the flag; `false` omits it.

```yaml
bfloat16:
  type: flag   # true → --bfloat16   false → (omitted)

tf32:
  type: flag   # true → --tf32       false → (omitted)
```

!!! note
    Use unquoted YAML `true`/`false` for `flag` and for boolean values in `categorical.choices`.
    Use **quoted** `"true"`/`"false"` when the wrapped script expects the literal string as a value (e.g. `--amp true`).

##### `group`

Bundles several CLI arguments together under a single Optuna categorical parameter. Optuna picks one group name; `iterate2` then injects all key/value pairs from that group into the trial's argument list. This is useful when multiple arguments are co-dependent (e.g. config file + dataset path + experiment name).

```yaml
dataset:
  type: group
  choices:
    case2000:
      config: ./examples/config/model_case2000.yaml
      data_path: /data/pf/
      exp_name: case2000
    case1000:
      config: ./examples/config/model_case1000.yaml
      data_path: /data/pf/
      exp_name: case1000
```

Optuna tracks the choice as a single categorical (`dataset = "case2000"`), but the wrapped script receives:

```
--config ./examples/config/model_case2000.yaml --data-path /data/pf/ --exp-name case2000
```

##### `gpu_num` — dynamic GPU count

The special key `gpu_num` (as `categorical` or `int`) is automatically extracted
from the sampled parameters and forwarded to the WLM plugin as
`ITERATE_WLM_GPU_COUNT`.  It does **not** appear in the wrapped script's command
line.  The WLM plugin uses it to set the cluster resource request for the trial.

```yaml
gpu_num:
  type: categorical
  choices: [1, 2, 4]
```

Alternatively, set a fixed `gpu-count` in the `wlm:` section of the HPO YAML
when all trials use the same number of GPUs.

### Static arguments

Arguments passed unchanged to every trial. Can be supplied inline or via file:

| Option | Description |
|---|---|
| `--static-args-json` | Static arguments as a JSON string |
| `--static-args-yaml` | Path to a YAML file with static arguments |

If neither is provided, `iterate2` falls back to the `static` section of `--hpo-yaml`.

Static boolean values follow the same rule as HPO values: unquoted `true` produces a bare flag (`--flag`), unquoted `false` omits it.

```yaml
static:
  max_epochs: 50
  tf32: true      # → --tf32  (store_true flag, always present)
  debug: false    # → (omitted)
```

### Metric extraction

| Option | Default | Description |
|---|---|
| `--metrics` | `score_combined` | Comma-separated list of metric names to extract from the trial's stdout log |

The **last** occurrence of the pattern `<metric_name>: <value>` or `<metric_name>= <value>` is used for each metric. If a metric is not found, it defaults to `0.0` with a warning.

**Single metric (single-objective):**

```sh
--metrics val_loss
```

**Multiple metrics (multi-objective, Pareto front):**

```sh
--metrics score_linear_acc,score_modality_leak,score_combined
```

`iterate2` prints the Pareto-front trials at the end:

```
Trial 12: Values=[0.873, 0.041, 0.791]
Trial 17: Values=[0.901, 0.038, 0.812]
```

#### Metric directions (minimize / maximize)

By default the direction per metric is inferred from its name: any metric whose name contains `loss`, `error`, `err`, `mse`, `mae`, or `rmse` is **minimized**; all others are **maximized**.

To override the direction explicitly, use the extended form in the `metrics:` section of the HPO YAML:

```yaml
metrics:
  - name: val_loss       # minimize (also inferred automatically)
    direction: minimize
  - name: accuracy       # maximize (also inferred automatically)
    direction: maximize
  - name: f1_score       # maximize (explicit override)
    direction: maximize
```

A mix of simple strings and extended objects is allowed:

```yaml
metrics:
  - val_loss             # direction inferred → minimize
  - name: my_custom_score
    direction: maximize  # direction explicit
```

---

## Setter-style arguments

Some scripts (e.g. those using [Hydra](https://hydra.cc/) overrides or custom key-value CLIs) do not accept named flags like `--learning-rate 0.001`. Instead they expect:

```sh
--set learning_rate 0.001 --set batch_size 32
```

Pass `--param-setter set` (or whatever flag name the target script uses) to switch `iterate2` to this style:

```sh
iterate2 --param-setter set ...
```

| Mode | Generated argument style |
|---|---|
| default (`--param-setter` omitted) | `--learning-rate 0.001 --batch-size 32` |
| `--param-setter set` | `--set learning_rate 0.001 --set batch_size 32` |

!!! note
    In setter style, `flag` parameters are passed as `--set flag` (key only, no value) when `true`, and omitted when `false`.

---

## WLM plugin system

iteate2 has no built-in knowledge of any workload manager.  Instead it calls a
user-supplied **plugin script** once per trial.  The plugin can be any
executable (bash, Python, …).

### Plugin interface

iterate2 calls the plugin with no positional arguments.  All information is
delivered through environment variables:

| Variable | Description |
|---|---|
| `ITERATE_TRIAL_NUMBER` | Integer trial ID |
| `ITERATE_TRIAL_CMD` | Full shell command (with `cd`, `source venv`) – suited for HPC WLMs |
| `ITERATE_TRIAL_CONTAINER_CMD` | Bare CLI invocation (no `cd`/`source`) – suited for container-based systems |
| `ITERATE_OUT_FILE` | File where **stdout** must be written |
| `ITERATE_ERR_FILE` | File where **stderr** must be written |
| `ITERATE_WLM_<KEY>` | Every key from the YAML `wlm:` section (uppercased, hyphens → underscores) |

The plugin must exit **0** on success; any other exit code marks the trial as
failed in Optuna.

### WLM configuration in the HPO YAML

All WLM-specific parameters (GPU count, memory, queue, job template path, …)
live in an optional `wlm:` section of the HPO YAML:

```yaml
hpo:
  lr: { type: float, low: 1e-5, high: 1e-2, log: true }

static:
  epochs: 50

# WLM config – forwarded as ITERATE_WLM_* env vars to the plugin
wlm:
  gpu-count: 1
  cpu-count: 8
  mem-gb: 32
  lsf-gpu-config: "num=1:mode=exclusive_process:mps=no:gmodel=NVIDIAA100_SXM4_80GB"
```

### Reference plugins

See `examples/wlm_plugins/` for fully documented reference implementations:

| Plugin | WLM |
|---|---|
| `lsf_plugin.sh` | IBM Spectrum LSF (`bsub -K`) |
| `vela_plugin.py` | OpenShift / MLBatch PyTorchJob (`helm template \| oc create`) |

Writing a SLURM plugin follows the same pattern as `lsf_plugin.sh`.

---

## Parallel execution

By default `iterate2` runs one trial at a time. Pass `--parallelism N` to run up to `N` trials simultaneously, each in its own thread.

```sh
iterate2 \
  --parallelism 4 \
  --wlm-plugin examples/wlm_plugins/lsf_plugin.sh \
  ...
```

### How it works

Each thread independently:

1. Asks Optuna for a new set of hyperparameters (`study.ask()`)
2. Builds and submits the launcher command (e.g. `bsub -K …`)
3. Streams every output line to the main process stdout/stderr, prefixed with `[trial-N]`
4. Reports the extracted metrics back to Optuna (`study.tell()`)

Output from concurrent trials is prefixed so you can follow individual workers:

```
[trial-3] Epoch 1/10  ━━━━━━━━━━ 100/100 0:01:12
[trial-5] Using bfloat16 precision
[trial-3] [performance] val_loss : 0.0421
[trial-5] Epoch 1/10  ━━━━━━━━━━ 100/100 0:01:15
```

### Output files

iteate2 tells the plugin where to write output via `ITERATE_OUT_FILE` /
`ITERATE_ERR_FILE`.  The plugin is responsible for directing its job's
stdout/stderr to those files.  iterate2 extracts metrics from them after the
plugin exits.

For local execution (no plugin) iterate2 writes them directly:

### SQLite and parallelism

Optuna retries on SQLite locking errors automatically. Values up to `--parallelism 4` work well with SQLite. For higher concurrency use PostgreSQL or **JournalStorage**:

```sh
# PostgreSQL
--optuna-db-path postgresql://user:pass@host/dbname

# JournalStorage (file-based, lock-free, safe for parallel workers on a shared filesystem)
--optuna-db-path js:///path/to/study_journal.log
```

`js:///` is a custom `iterate2` scheme. The path after `js:///` is passed to Optuna's `JournalFileStorage`. JournalStorage serialises trials to an append-only log and is well-suited for NFS/GPFS shared filesystems where SQLite locking is unreliable.

---

## Workload managers

### LSF

`iterate2` waits for each `bsub` job synchronously (`-K` flag). Output is written to `trial_<N>.out` / `trial_<N>.err` in the working directory.

### Slurm

Uses `srun` with `--gres=gpu:<N>`, `--cpus-per-task`, and `--mem` flags.

### none

Runs the command directly in a local shell, redirecting stdout/stderr to `trial_<N>.out` / `trial_<N>.err`.

### Vela (OpenShift / MLBatch)

`--wlm vela` submits each trial as a [PyTorchJob](https://www.kubeflow.org/docs/components/training/pytorch/) via the [MLBatch `pytorchjob-generator`](https://github.com/project-codeflare/mlbatch) helm chart.

#### Submission flow

1. For each Optuna trial iterate2:
    * Builds the CLI invocation from sampled + static args.
    * Patches the **job template YAML**:
        * appends `-trial-<N>` to `jobName` (unique resource per trial)
        * sets `numGpusPerPod` from `gpu_num` (HPO or CLI `--gpu-count`)
        * replaces the `{{HPO_COMMAND}}` placeholder in `setupCommands` with the generated CLI call
    * Runs `helm template -f <patched.yaml> <chart> | oc create [-n <ns>] -f-`
2. Polls until `<jobName>-master-0` pod appears, then streams `oc logs -f <pod>` — **this call blocks until the container exits**, so the trial behaves the same as other WLM backends.
3. Pod exit code is checked; non-zero raises an error.
4. The `PyTorchJob` resource is deleted.

#### Job template

Create a YAML file modelled on `examples/vela_gridfm_template.yaml`.  The only special requirement is the `{{HPO_COMMAND}}` placeholder somewhere in `setupCommands`:

```yaml
jobName: "my-project-hpo"       # iterate2 appends -trial-N
numGpusPerPod: 1                # iterate2 overwrites with gpu_num
numCpusPerPod: 32
totalMemoryPerPod: "32Gi"

volumes:
  - name: "data-vol"
    claimName: "my-pvc"
    mountPath: "/mnt/data"

setupCommands:
  - "wget -q https://example.com/config.yaml"
  - "{{HPO_COMMAND}}"           # ← iterate2 fills this in
```

#### Example invocation

```sh
iterate2 \
  --script            "gridfm_graphkit train" \
  --interpreter       "" \
  --wlm               vela \
  --vela-job-template examples/vela_gridfm_template.yaml \
  --vela-chart-path   ../mlbatch/tools/pytorchjob-generator/chart \
  --vela-namespace    my-namespace \
  --gpu-count         1 \
  --optuna-study-name gridfm_vela_hpo \
  --optuna-db-path    sqlite:///gridfm_vela_hpo.db \
  --optuna-n-trials   20 \
  --hpo-yaml          configs/gridfm_graphkit_hpo.yaml
```

See `examples/run_vela_example.sh` for a complete ready-to-run script.

!!! note
    `--script` is the bare CLI entry-point (`gridfm_graphkit train`).  Set `--interpreter ""` to suppress the default `python` prefix.

!!! tip
    `gpu_num` in the HPO space controls both `numGpusPerPod` in the job YAML **and** the WLM resource request, just like with LSF/Slurm.

---

## Example HPO YAML

Full example combining all parameter types:

```yaml
# hpo_space.yaml
hpo:
  # float – log-uniform over [1e-5, 1e-2]
  learning_rate:
    type: float
    low: 1e-5
    high: 1e-2
    log: true

  # int – encoder depth
  encoder_depth:
    type: int
    low: 2
    high: 8

  # categorical – batch size
  batch_size:
    type: categorical
    choices: [16, 32, 64]

  # categorical with null – compile mode (null omits --compile entirely)
  compile:
    type: categorical
    choices: ["max-autotune", "default", null]

  # flag – store_true style (--bfloat16 present or absent)
  bfloat16:
    type: flag

  # flag – store_true style (--tf32 present or absent)
  tf32:
    type: flag

  # gpu_num – controls WLM resource request per trial (not forwarded to script)
  gpu_num:
    type: categorical
    choices: [1, 2, 4]

  # group – bundles co-dependent args; Optuna picks one group by name
  dataset:
    type: group
    choices:
      case2000:
        config: ./examples/config/model_case2000.yaml
        data_path: /data/pf/
        exp_name: case2000
      case1000:
        config: ./examples/config/model_case1000.yaml
        data_path: /data/pf/
        exp_name: case1000

static:
  max_epochs: 50
  log_dir: logs
  num_workers: 16
```

Launch with:

```sh
iterate2 \
  --script gridfm_graphkit \
  --interpreter "" \
  --root-dir /path/to/project \
  --venv /path/to/venv \
  --wlm lsf \
  --lsf-gpu-config-string "num=1:mode=exclusive_process:mps=yes:gmodel=NVIDIAA100_SXM4_80GB" \
  --cpu-count 16 \
  --mem-gb 64 \
  --optuna-study-name my_study \
  --optuna-db-path sqlite:///my_study.db \
  --optuna-n-trials 50 \
  --hpo-yaml hpo_space.yaml \
  --metrics val_loss,val_f1
```
