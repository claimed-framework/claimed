#!/usr/bin/env python3

import argparse
import json
import logging
import os
import subprocess
import sys
import re
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List

import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import yaml

logger = logging.getLogger("iterate2")

# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic Optuna HPO launcher with Multi-Metric support"
    )

    # ------------------------
    # Execution config
    # ------------------------
    parser.add_argument("--script", required=True, help="Training script to execute")
    parser.add_argument("--root-dir", default=None, help="Root dir (derived if omitted)")
    parser.add_argument("--venv", default=".venv", help="Virtualenv dir")
    parser.add_argument("--interpreter", default="python", help="Interpreter to use")
    parser.add_argument("--param-setter", type=str, default=None)
    parser.add_argument("--wlm", choices=["lsf", "slurm", "openshift", "none"], default="none")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--cpu-count", type=int, default=4)
    parser.add_argument("--mem-gb", type=int, default=128)
    parser.add_argument("--lsf-gpu-config-string", type=str, default=None)
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of trials to run in parallel (default: 1 = sequential). "
             "Each parallel trial runs in its own thread. "
             "For SQLite storage, values >4 may cause locking contention; "
             "consider PostgreSQL for high parallelism.",
    )
    parser.add_argument(
        "--no-underscore-to-hyphen",
        dest="underscore_to_hyphen",
        action="store_false",
        default=True,
        help="Do not convert underscores to hyphens in arg names (default: convert)",
    )

    # ------------------------
    # Optuna config
    # ------------------------
    parser.add_argument("--optuna-study-name", required=True)
    parser.add_argument("--optuna-db-path", required=True)
    parser.add_argument("--optuna-n-trials", type=int, default=100)

    # ------------------------
    # HPO space
    # ------------------------
    parser.add_argument("--hpo-json", type=str, default=None)
    parser.add_argument("--hpo-yaml", type=str, default=None)
    parser.add_argument("--static-args-json", type=str, default=None)
    parser.add_argument("--static-args-yaml", type=str, default=None)

    # ------------------------
    # Metric extraction (Supports comma-separated list)
    # ------------------------
    parser.add_argument(
        "--metrics",
        default="score_combined",
        help="Comma-separated metric names to extract (e.g. score_linear_acc,score_modality_leak,score_combined)",
    )

    # ------------------------
    # Logging
    # ------------------------
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    return parser.parse_args()


# ============================================================
# HELPERS & COMMAND BUILDERS
# ============================================================

def resolve_paths(script: str, root_dir: Optional[str]):
    if root_dir is None: root_dir = '.'
    resolved = Path(root_dir).resolve()
    logger.debug("Resolved root_dir '%s' → '%s'", root_dir, resolved)
    return script, resolved

def build_launcher_command(wlm, cmd, trial_id, out_file, err_file, gpu_count, cpu_count, mem_gb, lsf_gpu_config_string):
    logger.debug("Building launcher command: wlm=%s gpu_count=%d cpu_count=%d mem_gb=%d", wlm, gpu_count, cpu_count, mem_gb)
    if wlm == "lsf":
        gpu_fragment = f"-gpu \"{lsf_gpu_config_string}\"" if lsf_gpu_config_string else (f"-gpu num={gpu_count}" if gpu_count > 0 else "")
        launcher = f"bsub {gpu_fragment} -K -o {out_file} -e {err_file} -R \"rusage[ngpus={gpu_count}, cpu={cpu_count}, mem={mem_gb}GB]\" -J hpo_trial_{trial_id} \"{cmd}\""
    elif wlm == "slurm":
        launcher = f"srun --gres=gpu:{gpu_count} --cpus-per-task={cpu_count} --mem={mem_gb}G --job-name=hpo_trial_{trial_id} --output={out_file} --error={err_file} bash -c \"{cmd}\""
    elif wlm == "none":
        # No embedded redirect: run_and_stream() captures stdout/stderr via PIPE
        # and writes to out_file/err_file itself.
        launcher = f'bash -c "{cmd}"'
    else:
        raise ValueError(f"Unknown WLM: {wlm}")
    logger.debug("Launcher command: %s", launcher)
    return launcher

def build_shell_command(interpreter, root_dir, script_path, venv, script_args, param_setter, underscore_to_hyphen=True):
    parts = [f"cd {root_dir}"]
    if venv:
        parts.append(f"source {venv}/bin/activate")
        logger.debug("Activating venv: %s", venv)
    arg_list = [f"{interpreter} {script_path}"]
    for key, value in script_args.items():
        arg_name = key.replace("_", "-") if underscore_to_hyphen else key
        if value is None:
            logger.debug("Skipping arg '%s': value is None (flag omitted)", key)
            continue
        if param_setter:
            if isinstance(value, bool):
                if value:
                    arg_list.append(f"--{param_setter} {key}")
                    logger.debug("Setter flag: --%s %s (store_true)", param_setter, key)
                else:
                    logger.debug("Skipping flag '%s': False → omitted", key)
            else:
                arg_list.append(f"--{param_setter} {key} {value}")
                logger.debug("Setter arg: --%s %s %s", param_setter, key, value)
        else:
            if isinstance(value, bool):
                if value:
                    arg_list.append(f"--{arg_name}")
                    logger.debug("Flag present: --%s", arg_name)
                else:
                    logger.debug("Skipping flag '--%s': False → omitted", arg_name)
            else:
                arg_list.append(f"--{arg_name} {value}")
                logger.debug("Arg: --%s %s", arg_name, value)
    cmd = " && ".join(parts + [" ".join(arg_list)])
    logger.debug("Shell command: %s", cmd)
    return cmd

# ============================================================
# PARALLEL STREAMING RUNNER
# ============================================================

_print_lock = threading.Lock()

def _stream_pipe(pipe, dest_file, trial_id: int, stream_name: str, dest_stream):
    """Read lines from *pipe*, write to *dest_file* and print prefixed to *dest_stream*."""
    prefix = f"[trial-{trial_id}]"
    with open(dest_file, "w", encoding="utf-8", errors="replace") as fh:
        for raw in pipe:
            line = raw.decode("utf-8", errors="replace")
            fh.write(line)
            fh.flush()
            with _print_lock:
                dest_stream.write(f"{prefix} {line}")
                dest_stream.flush()

def run_and_stream(launcher_cmd: str, trial_id: int, out_file: str, err_file: str, wlm: str):
    """
    Run *launcher_cmd* in a shell.

    For ``wlm='none'``: captures stdout and stderr via PIPE, streams every line
    to the main process stdout/stderr (prefixed with ``[trial-N]``), and also
    writes them to *out_file* / *err_file* for later metric extraction.

    For WLM backends (lsf, slurm, …): the WLM tool itself manages the output
    files on the cluster.  The local subprocess output (WLM status messages,
    errors) is still streamed with the same prefix so parallel workers are
    distinguishable.
    """
    logger.debug("Trial %d: run_and_stream wlm=%s cmd=%s", trial_id, wlm, launcher_cmd)
    proc = subprocess.Popen(
        launcher_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if wlm == "none":
        # Full capture: write to files AND stream to console
        t_out = threading.Thread(
            target=_stream_pipe,
            args=(proc.stdout, out_file, trial_id, "stdout", sys.stdout),
            daemon=True,
        )
        t_err = threading.Thread(
            target=_stream_pipe,
            args=(proc.stderr, err_file, trial_id, "stderr", sys.stderr),
            daemon=True,
        )
    else:
        # WLM manages the cluster output files (out_file/err_file) itself.
        # Stream only the local WLM tool output (bsub/srun status messages)
        # to console; write it to separate local files to avoid clobbering the
        # cluster-managed trial output files.
        wlm_out = out_file.replace(".out", "_wlm.out")
        wlm_err = err_file.replace(".err", "_wlm.err")
        t_out = threading.Thread(
            target=_stream_pipe,
            args=(proc.stdout, wlm_out, trial_id, "wlm-stdout", sys.stdout),
            daemon=True,
        )
        t_err = threading.Thread(
            target=_stream_pipe,
            args=(proc.stderr, wlm_err, trial_id, "wlm-stderr", sys.stderr),
            daemon=True,
        )

    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join()
    t_err.join()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, launcher_cmd)

# ============================================================
# MULTI-METRIC EXTRACTION
# ============================================================

def extract_metrics_from_log(path: str, metric_names: List[str], err_path: Optional[str] = None) -> List[float]:
    logger.debug("Extracting metrics %s from '%s'", metric_names, path)
    results = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    logger.debug("Log file '%s': %d characters read", path, len(text))
    # Also read stderr — Lightning/rich writes test result tables there
    if err_path:
        try:
            with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
                err_text = f.read()
            logger.debug("Err file '%s': %d characters read", err_path, len(err_text))
            text = text + "\n" + err_text
        except FileNotFoundError:
            logger.debug("Err file '%s' not found, skipping", err_path)

    for metric in metric_names:
        # Matches: key: value | key=value | [performance] key : value | Lightning table │ key │ value │
        pattern = re.compile(
            rf"(?:\[\w+\]\s*)?{re.escape(metric)}\s*(?:[:=│])\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        )
        matches = pattern.findall(text)
        if not matches:
            logger.warning("Metric '%s' not found in '%s' — defaulting to 0.0", metric, path)
            results.append(0.0)
        else:
            value = float(matches[-1])
            logger.debug("Metric '%s': found %d match(es), using last value %s", metric, len(matches), value)
            results.append(value)
    return results

# ============================================================
# MAIN
# ============================================================

def load_hpo_space(args):
    data = {}
    if args.hpo_json:
        logger.debug("Loading HPO space from JSON string")
        data = json.loads(args.hpo_json)
    elif args.hpo_yaml:
        logger.debug("Loading HPO space from YAML file: %s", args.hpo_yaml)
        with open(args.hpo_yaml, "r") as f: data = yaml.safe_load(f)
    space = data.get("hpo", {})
    logger.info("HPO space loaded: %d parameter(s): %s", len(space), list(space.keys()))
    return space

def load_metrics_from_yaml(args):
    """Return metrics list from YAML 'metrics:' key, or None if not present."""
    data = {}
    if args.hpo_json:
        data = json.loads(args.hpo_json)
    elif args.hpo_yaml:
        with open(args.hpo_yaml, "r") as f: data = yaml.safe_load(f)
    elif args.static_args_yaml:
        with open(args.static_args_yaml, "r") as f: data = yaml.safe_load(f)
    metrics = data.get("metrics", None)
    if metrics is None:
        return None
    if isinstance(metrics, list):
        return [m.strip() for m in metrics]
    return [m.strip() for m in str(metrics).split(",")]

def load_static_args(args):
    data = {}
    if args.static_args_json:
        logger.debug("Loading static args from JSON string")
        data = json.loads(args.static_args_json)
    elif args.static_args_yaml:
        logger.debug("Loading static args from YAML file: %s", args.static_args_yaml)
        with open(args.static_args_yaml, "r") as f: data = yaml.safe_load(f)
    elif args.hpo_yaml:
        logger.debug("Loading static args from HPO YAML file: %s", args.hpo_yaml)
        with open(args.hpo_yaml, "r") as f: data = yaml.safe_load(f)
    static = data.get("static", data if data else {})
    logger.info("Static args loaded: %d key(s): %s", len(static), list(static.keys()))
    return static

def suggest_from_spec(trial, name, spec):
    t = spec["type"]
    if t == "float":
        val = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=spec.get("log", False))
    elif t == "int":
        val = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=spec.get("log", False))
    elif t == "categorical":
        val = trial.suggest_categorical(name, spec["choices"])
    elif t == "flag":
        val = trial.suggest_categorical(name, [True, False])
    elif t == "group":
        val = trial.suggest_categorical(name, list(spec["choices"].keys()))
    else:
        raise ValueError(f"Unknown param type: {t}")
    logger.debug("Suggested '%s' (%s) = %r", name, t, val)
    return val

def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress noisy optuna INFO logs unless user asked for DEBUG
    logging.getLogger("optuna").setLevel(
        logging.WARNING if args.log_level == "INFO" else getattr(logging, args.log_level)
    )

    logger.info("iterate2 starting")
    logger.info("Log level: %s", args.log_level)
    logger.info("WLM: %s | interpreter: %s | script: %s", args.wlm, args.interpreter, args.script)
    logger.info("Optuna study: '%s' | db: %s | n_trials: %d", args.optuna_study_name, args.optuna_db_path, args.optuna_n_trials)

    hpo_space = load_hpo_space(args)
    static_args = load_static_args(args)
    yaml_metrics = load_metrics_from_yaml(args)
    metric_list = yaml_metrics if yaml_metrics is not None else [m.strip() for m in args.metrics.split(",")]
    logger.info("Optimising metrics: %s (source: %s)", metric_list, "yaml" if yaml_metrics else "cli")

    script_path, root_dir = resolve_paths(args.script, args.root_dir)

    def objective(trial):
        script_args = static_args.copy()
        for name, spec in hpo_space.items():
            val = suggest_from_spec(trial, name, spec)
            if spec["type"] == "group":
                # Expand the chosen group's key→value pairs directly into script_args
                script_args.update(spec["choices"][val])
            else:
                script_args[name] = val

        # gpu_num in hpo/static overrides the CLI --gpu-count for this trial's launcher
        gpu_count = int(script_args.pop("gpu_num", args.gpu_count))
        logger.debug("Trial %d: effective gpu_count=%d", trial.number, gpu_count)
        logger.info("Trial %d: sampled parameters: %s", trial.number, script_args)

        out_file = f"trial_{trial.number}.out"
        err_file = f"trial_{trial.number}.err"
        logger.debug("Trial %d: stdout → %s | stderr → %s", trial.number, out_file, err_file)

        shell_cmd = build_shell_command(args.interpreter, root_dir, script_path, args.venv, script_args, args.param_setter, args.underscore_to_hyphen)
        launcher_cmd = build_launcher_command(args.wlm, shell_cmd, trial.number, out_file, err_file, gpu_count, args.cpu_count, args.mem_gb, args.lsf_gpu_config_string)

        logger.info("Trial %d: submitting → %s", trial.number, launcher_cmd)
        run_and_stream(launcher_cmd, trial.number, out_file, err_file, args.wlm)
        logger.info("Trial %d: job finished", trial.number)

        values = extract_metrics_from_log(out_file, metric_list, err_path=err_file)
        logger.info("Trial %d: results %s", trial.number, dict(zip(metric_list, values)))

        return tuple(values)

    # Multi-objective direction
    directions = ["maximize"] * len(metric_list)
    logger.info("Creating Optuna study (directions: %s)", directions)

    if args.optuna_db_path.startswith("js:///"):
        journal_path = args.optuna_db_path[len("js://"):]
        logger.info("Using JournalStorage at '%s'", journal_path)
        storage = JournalStorage(JournalFileStorage(journal_path))
    elif "sqlite" in args.optuna_db_path:
        storage = args.optuna_db_path
    else:
        storage = f"sqlite:///{args.optuna_db_path}"
    logger.debug("Optuna storage: %s", storage)

    study = optuna.create_study(
        study_name=args.optuna_study_name,
        storage=storage,
        directions=directions,
        load_if_exists=True,
    )
    logger.info("Study '%s' ready (existing trials: %d)", args.optuna_study_name, len(study.trials))

    logger.info("Parallelism: %d worker(s)", args.parallelism)
    study.optimize(objective, n_trials=args.optuna_n_trials, n_jobs=args.parallelism)

    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("Pareto Front Trials: %d", len(study.best_trials))
    for t in study.best_trials:
        logger.info("  Trial %d: Values=%s  Params=%s", t.number, t.values, t.params)

if __name__ == "__main__":
    main()