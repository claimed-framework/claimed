#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, Optional, Literal

import optuna
import yaml

# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic Optuna HPO launcher with pluggable execution backend"
    )

    # ------------------------
    # Execution config
    # ------------------------
    parser.add_argument("--script", required=True, help="Training script to execute")
    parser.add_argument("--root-dir", default=None, help="Root dir (derived if omitted)")
    parser.add_argument("--venv", default=".venv", help="Virtualenv dir, default: .venv (set empty to disable)")
    parser.add_argument("--interpreter", default="python", help="Interpreter to use, default: python")
    parser.add_argument(
        "--param-setter",
        type=str,
        default=None,
        help=(
            "When the target script uses a setter-style interface instead of named flags, "
            "provide the setter flag name here (e.g. 'set'). "
            "Each parameter will then be passed as '--<setter> key value' "
            "instead of the default '--key value' style. "
            "Example: --param-setter set  →  --set learning_rate 0.001 --set batch_size 32"
        ),
    )
    parser.add_argument(
        "--wlm",
        choices=["lsf", "slurm", "openshift", "none"],
        default="none",
        help="Workload manager",
    )
    parser.add_argument("--gpu-count", type=int, default=1, help="GPUs per trial")
    parser.add_argument("--cpu-count", type=int, default=4, help="CPUs per trial")
    parser.add_argument("--mem-gb", type=int, default=128, help="Memory (GB) per trial")
    parser.add_argument(
        "--lsf-gpu-config-string",
        type=str,
        default=None,
        help=(
            "Optional LSF -gpu option string, e.g. "
            "'num=1:mode=exclusive_process:mps=yes:gmodel=NVIDIAA100_SXM4_80GB'. "
            "When set, it is passed verbatim as -gpu \"<string>\" in the bsub command "
            "and overrides the default --gpu-count-based -gpu flag."
        ),
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
    parser.add_argument(
        "--hpo-json",
        type=str,
        default=None,
        help="HPO search space as JSON string",
    )
    parser.add_argument(
        "--hpo-yaml",
        type=str,
        default=None,
        help="HPO search space YAML file",
    )

    # ------------------------
    # Static arguments (passed to every trial)
    # ------------------------
    parser.add_argument(
        "--static-args-json",
        type=str,
        default=None,
        help="Static arguments as JSON string (key-value pairs)",
    )
    parser.add_argument(
        "--static-args-yaml",
        type=str,
        default=None,
        help="Static arguments YAML file (key-value pairs)",
    )

    # ------------------------
    # Metric extraction
    # ------------------------
    parser.add_argument(
        "--metric",
        default="val/F1_Score",
        help="Metric name to extract from logs",
    )

    return parser.parse_args()


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_paths(script: str, root_dir: Optional[str]):
    script_path = Path(script).resolve()

    if root_dir is None:
        root_dir = '.'

    return script, Path(root_dir).resolve()


# ============================================================
# WORKLOAD MANAGER ABSTRACTION
# ============================================================

def build_launcher_command(
    *,
    wlm: Literal["lsf", "slurm", "openshift", "none"],
    cmd: str,
    trial_id: int,
    out_file: str,
    err_file: str,
    gpu_count: int = 1,
    cpu_count: int = 4,
    mem_gb: int = 128,
    lsf_gpu_config_string: Optional[str] = None,
):
    if wlm == "lsf":
        if lsf_gpu_config_string is not None:
            gpu_fragment = f"-gpu \"{lsf_gpu_config_string}\""
        elif gpu_count > 0:
            gpu_fragment = f"-gpu num={gpu_count}"
        else:
            gpu_fragment = ""

        if gpu_fragment:
            return (
                f"bsub {gpu_fragment} -K "
                f"-o {out_file} -e {err_file} "
                f"-R \"rusage[ngpus={gpu_count}, cpu={cpu_count}, mem={mem_gb}GB]\" "
                f"-J hpo_trial_{trial_id} "
                f"\"{cmd}\""
            )
        else:
            return (
                f"bsub -K "
                f"-o {out_file} -e {err_file} "
                f"-R \"rusage[cpu={cpu_count}, mem={mem_gb}GB]\" "
                f"-J hpo_trial_{trial_id} "
                f"\"{cmd}\""
            )

    if wlm == "slurm":
        return (
            f"srun --gres=gpu:{gpu_count} --cpus-per-task={cpu_count} --mem={mem_gb}G "
            f"--job-name=hpo_trial_{trial_id} "
            f"--output={out_file} --error={err_file} "
            f"bash -c \"{cmd}\""
        )

    if wlm == "openshift":
        raise NotImplementedError("OpenShift launcher not implemented yet")

    if wlm == "none":
        return f'bash -c "{cmd} > {out_file} 2> {err_file}"'

    raise ValueError(f"Unknown workload manager: {wlm}")


# ============================================================
# SHELL COMMAND BUILDER
# ============================================================

def build_shell_command(
    *,
    interpreter: str = "python",
    root_dir: Path,
    script_path: Path,
    venv: Optional[str],
    script_args: Dict[str, Any],
    param_setter: Optional[str] = None,
):
    """
    Build shell command for argparse-based scripts.

    Args:
        root_dir: Root directory to cd into
        script_path: Path to the Python script
        venv: Virtual environment directory (optional)
        script_args: Dictionary of argument name -> value for the script
        param_setter: When set, use '--<setter> key value' style instead of
            '--key value'.  E.g. param_setter='set' produces
            '--set learning_rate 0.001' instead of '--learning-rate 0.001'.
    """
    parts = [f"cd {root_dir}"]

    if venv:
        parts.append(f"source {venv}/bin/activate")

    # Build the python command with arguments
    arg_list = [f"{interpreter} {script_path}"]

    for key, value in script_args.items():
        if key == "value_only":
            arg_list.append(str(value))
            continue

        if param_setter is not None:
            # Setter style: --<setter> key value  (booleans passed explicitly)
            bool_value = None
            if isinstance(value, bool):
                bool_value = value
            elif isinstance(value, str) and value.lower() in ("false", "true"):
                bool_value = value.lower() == "true"

            if bool_value is not None:
                arg_list.append(f"--{param_setter} {key} {str(bool_value).lower()}")
            else:
                str_value = str(value)
                if " " in str_value:
                    arg_list.append(f'--{param_setter} {key} "{str_value}"')
                else:
                    arg_list.append(f"--{param_setter} {key} {str_value}")
        else:
            # Default style: --key value
            arg_name = key.replace("_", "-")

            if isinstance(value, bool):
                if value:
                    arg_list.append(f"--{arg_name}")
            elif isinstance(value, str) and value.lower() in ("false", "true"):
                if value.lower() == "true":
                    arg_list.append(f"--{arg_name}")
            else:
                arg_list.append(f"--{arg_name} {value}")

    parts.append(" ".join(arg_list))

    return " && ".join(parts)


# ============================================================
# HPO SPACE LOADING
# ============================================================

def load_hpo_space(args) -> Dict[str, Any]:
    """Load HPO search space configuration."""
    data = {}
    if args.hpo_json:
        data = json.loads(args.hpo_json)
    elif args.hpo_yaml:
        with open(args.hpo_yaml, "r") as f:
            data = yaml.safe_load(f)
    
    if isinstance(data, dict) and "hpo" in data:
        return data["hpo"]
    return {}

def load_static_args(args) -> Dict[str, Any]:
    """Load static arguments."""
    data = {}
    if args.static_args_json:
        data = json.loads(args.static_args_json)
    elif args.static_args_yaml:
        with open(args.static_args_yaml, "r") as f:
            data = yaml.safe_load(f)
    elif args.hpo_yaml: # Fallback: check the HPO file for a 'static' section
        with open(args.hpo_yaml, "r") as f:
            data = yaml.safe_load(f)

    if isinstance(data, dict) and "static" in data:
        return data["static"]
    return data if data else {}


# ============================================================
# OPTUNA PARAM INSTANTIATION
# ============================================================

def _num(x):
    """Convert string to numeric if needed."""
    if isinstance(x, str):
        return float(x)
    return x


def suggest_from_spec(trial, name: str, spec: Dict[str, Any]):
    """
    Suggest a hyperparameter value based on specification.
    
    Args:
        trial: Optuna trial object
        name: Parameter name
        spec: Parameter specification dict with 'type' and other fields
    
    Returns:
        Suggested parameter value
    """
    t = spec["type"]

    if t == "float":
        return trial.suggest_float(
            name,
            _num(spec["low"]),
            _num(spec["high"]),
            log=spec.get("log", False),
        )

    if t == "int":
        return trial.suggest_int(
            name,
            int(_num(spec["low"])),
            int(_num(spec["high"])),
            log=spec.get("log", False),
        )

    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])

    raise ValueError(f"Unknown param type: {t}")


# ============================================================
# METRIC EXTRACTION
# ============================================================

def extract_metric_from_log(path, metric: str):
    pattern = re.compile(
        rf"{re.escape(metric)}\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    matches = pattern.findall(text)

    if not matches:
        raise RuntimeError(f"Metric '{metric}' not found in {path}")

    return float(matches[-1])


# ============================================================
# MAIN
# ============================================================

def main():
    print("IMPORTANT: For the old iterate HPO Launcher v0.2.3 use iterate-classic")
    args = parse_args()
    hpo_space = load_hpo_space(args)
    static_args = load_static_args(args)

    script_path, root_dir = resolve_paths(args.script, args.root_dir)

    def objective(trial):
        # Combine static arguments with HPO parameters
        script_args = static_args.copy()
        
        # Add HPO parameters
        for name, spec in hpo_space.items():
            script_args[name] = suggest_from_spec(trial, name, spec)

        # Add trial number
        #script_args["trial_number"] = trial.number

        # Log file paths
        out_file = f"trial_{trial.number}.out"
        err_file = f"trial_{trial.number}.err"

        # Build command
        shell_cmd = build_shell_command(
            root_dir=root_dir,
            interpreter=args.interpreter,
            script_path=script_path,
            venv=args.venv if args.venv else None,
            script_args=script_args,
            param_setter=args.param_setter,
        )

        launcher_cmd = build_launcher_command(
            wlm=args.wlm,
            cmd=shell_cmd,
            trial_id=trial.number,
            out_file=out_file,
            err_file=err_file,
            gpu_count=args.gpu_count,
            cpu_count=args.cpu_count,
            mem_gb=args.mem_gb,
            lsf_gpu_config_string=args.lsf_gpu_config_string,
        )

        # Execute trial
        print(f"Trial {trial.number}: Executing command")
        print(f"  HPO params: {hpo_space.keys()}")
        subprocess.run(launcher_cmd, shell=True, check=True)

        # Extract and return metric
        metric_value = extract_metric_from_log(out_file, args.metric)
        print(f"Trial {trial.number}: {args.metric} = {metric_value}")
        
        return metric_value

    # Create or load Optuna study
    study = optuna.create_study(
        study_name=args.optuna_study_name,
        storage=args.optuna_db_path,
        direction="maximize",
        load_if_exists=True,
    )

    # Run optimization
    study.optimize(objective, n_trials=args.optuna_n_trials)
    
    # Print best results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()