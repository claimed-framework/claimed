# iterate2 – HPO Launcher

`iterate2` is a generic Optuna-based hyperparameter optimisation (HPO) launcher with a pluggable workload-manager backend. It submits one trial per Optuna suggestion, waits for the job to finish, extracts one or more metrics from the job's log file, and returns them to Optuna.

Key capabilities:

- **Multi-objective optimisation** — extract and optimise several metrics simultaneously (Pareto front)
- **Five HPO parameter types** — `float`, `int`, `categorical`, `flag` (store-true), `group` (bundled arg sets)
- **Dynamic GPU count per trial** — `gpu_num` in the HPO space controls the WLM resource request per trial
- **Null-omission** — `null` in a `categorical` choice causes the flag to be completely absent from the command line
- **Workload manager backends** — LSF, Slurm, or direct local execution

## Quick start

```sh
iterate2 \
  --script train.py \
  --wlm lsf \
  --gpu-count 1 \
  --cpu-count 20 \
  --mem-gb 512 \
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
| `--venv` | `.venv` | Virtual-environment directory to activate. Set to empty string to disable |
| `--interpreter` | `python` | Python interpreter to invoke |
| `--param-setter` | `None` | Use setter-style argument passing (see [Setter-style arguments](#setter-style-arguments)) |
| `--wlm` | `none` | Workload manager: `lsf`, `slurm`, `openshift`, or `none` |
| `--gpu-count` | `1` | Number of GPUs per trial |
| `--cpu-count` | `4` | Number of CPUs per trial |
| `--mem-gb` | `128` | Memory (GB) per trial |
| `--lsf-gpu-config-string` | `None` | Optional verbatim LSF `-gpu` option string (see [GPU configuration](#gpu-configuration-on-lsf)) |
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

The special key `gpu_num` (as `categorical` or `int`) overrides `--gpu-count` for the **WLM resource request** of each individual trial. It is consumed by `iterate2` and never forwarded to the wrapped script.

```yaml
gpu_num:
  type: categorical
  choices: [1, 2, 4]
```

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

All objectives are maximised. `iterate2` prints the Pareto-front trials at the end:

```
Trial 12: Values=[0.873, 0.041, 0.791]
Trial 17: Values=[0.901, 0.038, 0.812]
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

## GPU configuration on LSF

When `--wlm lsf` is selected, `iterate2` constructs a `bsub` command for each trial.

### Default behaviour

| `--gpu-count` | Generated fragment |
|---|---|
| `> 0` (default `1`) | `-gpu num=<N>` |
| `0` | *(no `-gpu` flag, CPU-only job)* |

### `--lsf-gpu-config-string`

For advanced LSF GPU scheduling you can supply the full value of the `-gpu` option as a string. When set, it **completely replaces** the auto-generated `-gpu num=<N>` fragment.

```sh
iterate2 \
  --wlm lsf \
  --lsf-gpu-config-string "num=1:mode=exclusive_process:mps=yes:gmodel=NVIDIAA100_SXM4_80GB" \
  --cpu-count 20 \
  --mem-gb 512 \
  ...
```

This produces a `bsub` submission resembling:

```sh
bsub -n 20 -R "span[hosts=1]" \
     -gpu "num=1:mode=exclusive_process:mps=yes:gmodel=NVIDIAA100_SXM4_80GB" \
     -M 512G -J hpo_trial_0 \
     "cd /my/root && source .venv/bin/activate && python train.py ..."
```

!!! note
    `--gpu-count` is still used for the `rusage` memory/CPU reservation string even when `--lsf-gpu-config-string` is set. Set it to match the `num=` value in your GPU string.

!!! tip
    Use exclusive process mode (`mode=exclusive_process`) together with MPS (`mps=yes`) to share a single A100 across multiple MPS clients while still pinning the job to one physical GPU.

---

---

## Parallel execution

By default `iterate2` runs one trial at a time. Pass `--parallelism N` to run up to `N` trials simultaneously, each in its own thread.

```sh
iterate2 \
  --parallelism 4 \
  --wlm lsf \
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

| WLM | stdout | stderr |
|---|---|---|
| `none` | `trial_N.out` (written by iterate2) | `trial_N.err` (written by iterate2) |
| `lsf` / `slurm` | `trial_N.out` (written by WLM on cluster) | `trial_N.err` (written by WLM on cluster) |

For WLM backends the local WLM tool output (bsub/srun status messages) is written to `trial_N_wlm.out` / `trial_N_wlm.err` so the cluster-managed files are never overwritten.

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

### openshift

Not yet implemented.

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
