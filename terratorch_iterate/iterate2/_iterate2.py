#!/usr/bin/env python3
"""
iterate2 – minimal Optuna HPO launcher.

iterate2 does exactly three things:
  1. Load the HPO search space and static parameters from a YAML file.
  2. For every Optuna trial, sample parameters and call a user-provided
     script with those parameters exposed as environment variables.
  3. After the script exits, extract one or more metrics from the log file
     the script wrote and return them to Optuna.

The user script is fully in charge of *how* the trial runs – activating a
virtualenv, submitting a bsub/sbatch job, running locally, launching a
container, etc.  iterate2 has no opinion on any of that.

Environment variables passed to the script
------------------------------------------
  ITERATE_TRIAL_NUMBER   integer trial ID (0-based)
  ITERATE_OUT_FILE       path the script must write its stdout to
  ITERATE_ERR_FILE       path the script must write its stderr to
  ITERATE_PARAM_<KEY>    one variable per sampled + static parameter
                         (key uppercased, hyphens and spaces → underscores)

HPO YAML format
---------------
  metrics:              # list of metric names to extract from ITERATE_OUT_FILE
    - val_loss
    - accuracy

  static:               # fixed parameters, forwarded as-is every trial
    epochs: 50
    dataset: /data/my_dataset

  hpo:                  # parameters Optuna will optimise
    learning_rate:
      type: float
      low: 1e-5
      high: 1e-2
      log: true
    batch_size:
      type: categorical
      choices: [16, 32, 64]
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

import optuna
import yaml
from terratorch_iterate.iterate2.plugin.coordinator import load_builtin_plugins, resolve_storage

load_builtin_plugins()

logger = logging.getLogger("iterate2")

# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Minimal Optuna HPO launcher – calls a user script per trial"
    )
    p.add_argument("--script",            required=True,
                   help="Executable to call for each trial")
    p.add_argument("--hpo-yaml",          required=True,
                   help="YAML file with 'hpo:', 'static:', and 'metrics:' sections")
    p.add_argument("--optuna-study-name", required=True)
    p.add_argument("--optuna-db-path",    required=True,
                   help="Optuna storage URL (sqlite:///hpo.db, js:///journal.log, postgresql://…)")
    p.add_argument("--optuna-n-trials",   type=int, default=100)
    p.add_argument("--parallelism",       type=int, default=1,
                   help="Parallel trials (threads). Use PostgreSQL/JournalStorage for >4.")
    p.add_argument("--log-level",         default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ─── YAML LOADING ────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}

def load_hpo_space(data: dict) -> dict:
    space = data.get("hpo", {})
    logger.info("HPO space: %d param(s): %s", len(space), list(space.keys()))
    return space

def load_static(data: dict) -> dict:
    static = data.get("static", {})
    logger.info("Static params: %d key(s): %s", len(static), list(static.keys()))
    return static

def load_metrics(data: dict, fallback: str = "score") -> List[str]:
    raw = data.get("metrics", None)
    if raw is None:
        logger.warning("No 'metrics:' key in YAML – defaulting to '%s'", fallback)
        return [fallback]
    if isinstance(raw, list):
        return [str(m).strip() for m in raw]
    return [m.strip() for m in str(raw).split(",")]


# ─── OPTUNA PARAM SAMPLING ───────────────────────────────────────────────────

def suggest(trial: optuna.Trial, name: str, spec: dict):
    t = spec["type"]
    if t == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                                   log=spec.get("log", False))
    if t == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]),
                                 log=spec.get("log", False))
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if t == "flag":
        return trial.suggest_categorical(name, [True, False])
    if t == "group":
        return trial.suggest_categorical(name, list(spec["choices"].keys()))
    raise ValueError(f"Unknown param type '{t}' for '{name}'")


# ─── METRIC EXTRACTION ───────────────────────────────────────────────────────

def extract_metrics(out_file: str, err_file: str, metric_names: List[str]) -> List[float]:
    """Read both output files and extract the last occurrence of each metric."""
    text = ""
    for path in (out_file, err_file):
        try:
            text += Path(path).read_text(encoding="utf-8", errors="ignore") + "\n"
        except FileNotFoundError:
            pass

    results = []
    for metric in metric_names:
        # Support  name#N  for Nth-occurrence selection (0-based)
        occurrence: Optional[int] = None
        bare = metric
        m = re.fullmatch(r'(.+)#(\d+)', metric)
        if m:
            bare, occurrence = m.group(1), int(m.group(2))

        pattern = re.compile(
            rf"(?:\[\w+\]\s*)?{re.escape(bare)}\s*[:=│]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        )
        matches = pattern.findall(text)
        if not matches:
            logger.warning("Metric '%s' not found – defaulting to 0.0", metric)
            results.append(0.0)
        elif occurrence is not None:
            if occurrence >= len(matches):
                logger.warning("Metric '%s' occurrence #%d not found – defaulting to 0.0",
                               metric, occurrence)
                results.append(0.0)
            else:
                results.append(float(matches[occurrence]))
        else:
            results.append(float(matches[-1]))
    return results


# ─── SCRIPT RUNNER ───────────────────────────────────────────────────────────

_print_lock = threading.Lock()

def _stream(pipe, dest_file: str, trial_id: int, dest_stream):
    prefix = f"[trial-{trial_id}]"
    with open(dest_file, "w", encoding="utf-8", errors="replace") as fh:
        for raw in pipe:
            line = raw.decode("utf-8", errors="replace")
            fh.write(line)
            fh.flush()
            with _print_lock:
                dest_stream.write(f"{prefix} {line}")
                dest_stream.flush()

def run_script(script: str, env: dict, trial_id: int, out_file: str, err_file: str):
    """Run *script* with *env*, stream output, raise on non-zero exit."""
    logger.info("Trial %d: calling %s", trial_id, script)
    proc = subprocess.Popen(
        [script], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    import threading as _t
    t_out = _t.Thread(target=_stream, args=(proc.stdout, out_file,  trial_id, sys.stdout), daemon=True)
    t_err = _t.Thread(target=_stream, args=(proc.stderr, err_file, trial_id, sys.stderr), daemon=True)
    t_out.start(); t_err.start()
    proc.wait()
    t_out.join(); t_err.join()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, script)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("optuna").setLevel(
        logging.WARNING if args.log_level == "INFO" else getattr(logging, args.log_level)
    )

    logger.info("iterate2 starting  script=%s  yaml=%s", args.script, args.hpo_yaml)
    logger.info("study=%s  db=%s  n_trials=%d  parallelism=%d",
                args.optuna_study_name, args.optuna_db_path,
                args.optuna_n_trials, args.parallelism)

    data       = load_yaml(args.hpo_yaml)
    hpo_space  = load_hpo_space(data)
    static     = load_static(data)
    metrics    = load_metrics(data)
    directions = ["maximize"] * len(metrics)
    logger.info("Metrics: %s", metrics)

    storage = resolve_storage(args.optuna_db_path)
    study = optuna.create_study(
        study_name=args.optuna_study_name,
        storage=storage,
        directions=directions,
        load_if_exists=True,
    )
    logger.info("Study '%s' ready (existing trials: %d)",
                args.optuna_study_name, len(study.trials))

    # ── Re-queue failed trials: 25 % retry, 75 % new ─────────────────────
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    n_total = args.optuna_n_trials
    if failed:
        n_retry = min(max(1, round(0.25 * n_total)), len(failed))
        n_new   = n_total - n_retry
        logger.info("Found %d failed trial(s) – re-queuing %d (25%%), %d new (75%%)",
                    len(failed), n_retry, n_new)
        for ft in failed[-n_retry:]:
            if ft.params:
                study.enqueue_trial(ft.params)
                logger.info("  Enqueued failed trial %d params: %s", ft.number, ft.params)
    # ─────────────────────────────────────────────────────────────────────

    def objective(trial):
        # ── Sample parameters ─────────────────────────────────────────────
        params = dict(static)  # start with static params
        for name, spec in hpo_space.items():
            val = suggest(trial, name, spec)
            if spec["type"] == "group":
                params.update(spec["choices"][val])  # expand group → flat keys
            else:
                params[name] = val

        logger.info("Trial %d params: %s", trial.number, params)

        out_file = f"trial_{trial.number}.out"
        err_file = f"trial_{trial.number}.err"

        # ── Build env for the script ──────────────────────────────────────
        env = os.environ.copy()
        env["ITERATE_TRIAL_NUMBER"] = str(trial.number)
        env["ITERATE_OUT_FILE"]     = out_file
        env["ITERATE_ERR_FILE"]     = err_file
        for k, v in params.items():
            env_key = "ITERATE_PARAM_" + str(k).upper().replace("-", "_").replace(" ", "_")
            env[env_key] = str(v) if v is not None else ""

        # ── Call the script ───────────────────────────────────────────────
        run_script(args.script, env, trial.number, out_file, err_file)

        # ── Extract metrics ───────────────────────────────────────────────
        values = extract_metrics(out_file, err_file, metrics)
        logger.info("Trial %d results: %s", trial.number, dict(zip(metrics, values)))
        return tuple(values)

    logger.info("Starting optimisation (%d worker(s))", args.parallelism)
    study.optimize(
        objective,
        n_trials=n_total,
        n_jobs=args.parallelism,
        catch=(Exception,),
    )

    logger.info("=" * 60)
    logger.info("OPTIMISATION COMPLETE  Pareto front: %d trial(s)", len(study.best_trials))
    for t in study.best_trials:
        logger.info("  Trial %d: values=%s  params=%s", t.number, t.values, t.params)


if __name__ == "__main__":
    main()
