#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List

import optuna
import yaml

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

    return parser.parse_args()


# ============================================================
# HELPERS & COMMAND BUILDERS
# ============================================================

def resolve_paths(script: str, root_dir: Optional[str]):
    if root_dir is None: root_dir = '.'
    return script, Path(root_dir).resolve()

def build_launcher_command(wlm, cmd, trial_id, out_file, err_file, gpu_count, cpu_count, mem_gb, lsf_gpu_config_string):
    if wlm == "lsf":
        gpu_fragment = f"-gpu \"{lsf_gpu_config_string}\"" if lsf_gpu_config_string else (f"-gpu num={gpu_count}" if gpu_count > 0 else "")
        return f"bsub {gpu_fragment} -K -o {out_file} -e {err_file} -R \"rusage[ngpus={gpu_count}, cpu={cpu_count}, mem={mem_gb}GB]\" -J hpo_trial_{trial_id} \"{cmd}\""
    if wlm == "slurm":
        return f"srun --gres=gpu:{gpu_count} --cpus-per-task={cpu_count} --mem={mem_gb}G --job-name=hpo_trial_{trial_id} --output={out_file} --error={err_file} bash -c \"{cmd}\""
    if wlm == "none":
        return f'bash -c "{cmd} > {out_file} 2> {err_file}"'
    raise ValueError(f"Unknown WLM: {wlm}")

def build_shell_command(interpreter, root_dir, script_path, venv, script_args, param_setter):
    parts = [f"cd {root_dir}"]
    if venv: parts.append(f"source {venv}/bin/activate")
    arg_list = [f"{interpreter} {script_path}"]
    for key, value in script_args.items():
        arg_name = key.replace("_", "-")
        if param_setter:
            arg_list.append(f"--{param_setter} {key} {value}")
        else:
            if isinstance(value, bool):
                if value: arg_list.append(f"--{arg_name}")
            else:
                arg_list.append(f"--{arg_name} {value}")
    parts.append(" ".join(arg_list))
    return " && ".join(parts)

# ============================================================
# MULTI-METRIC EXTRACTION
# ============================================================

def extract_metrics_from_log(path: str, metric_names: List[str]) -> List[float]:
    results = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    for metric in metric_names:
        pattern = re.compile(rf"{re.escape(metric)}\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
        matches = pattern.findall(text)
        if not matches:
            print(f"Warning: Metric '{metric}' not found in {path}. Defaulting to 0.0")
            results.append(0.0)
        else:
            results.append(float(matches[-1]))
    return results

# ============================================================
# MAIN
# ============================================================

def load_hpo_space(args):
    data = {}
    if args.hpo_json: data = json.loads(args.hpo_json)
    elif args.hpo_yaml:
        with open(args.hpo_yaml, "r") as f: data = yaml.safe_load(f)
    return data.get("hpo", {})

def load_static_args(args):
    data = {}
    if args.static_args_json: data = json.loads(args.static_args_json)
    elif args.static_args_yaml:
        with open(args.static_args_yaml, "r") as f: data = yaml.safe_load(f)
    elif args.hpo_yaml:
        with open(args.hpo_yaml, "r") as f: data = yaml.safe_load(f)
    return data.get("static", data if data else {})

def suggest_from_spec(trial, name, spec):
    t = spec["type"]
    if t == "float": return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=spec.get("log", False))
    if t == "int": return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=spec.get("log", False))
    if t == "categorical": return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown param type: {t}")

def main():
    args = parse_args()
    hpo_space = load_hpo_space(args)
    static_args = load_static_args(args)
    metric_list = [m.strip() for m in args.metrics.split(",")]
    
    script_path, root_dir = resolve_paths(args.script, args.root_dir)

    def objective(trial):
        script_args = static_args.copy()
        for name, spec in hpo_space.items():
            script_args[name] = suggest_from_spec(trial, name, spec)

        out_file = f"trial_{trial.number}.out"
        err_file = f"trial_{trial.number}.err"

        shell_cmd = build_shell_command(args.interpreter, root_dir, script_path, args.venv, script_args, args.param_setter)
        launcher_cmd = build_launcher_command(args.wlm, shell_cmd, trial.number, out_file, err_file, args.gpu_count, args.cpu_count, args.mem_gb, args.lsf_gpu_config_string)

        print(f"Trial {trial.number}: Running...")
        subprocess.run(launcher_cmd, shell=True, check=True)

        values = extract_metrics_from_log(out_file, metric_list)
        print(f"Trial {trial.number} results: {dict(zip(metric_list, values))}")
        
        return tuple(values)

    # Multi-objective direction
    directions = ["maximize"] * len(metric_list)

    study = optuna.create_study(
        study_name=args.optuna_study_name,
        storage=f"sqlite:///{args.optuna_db_path}" if "sqlite" not in args.optuna_db_path else args.optuna_db_path,
        directions=directions,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=args.optuna_n_trials)

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print(f"Pareto Front Trials: {len(study.best_trials)}")
    for t in study.best_trials:
        print(f"Trial {t.number}: Values={t.values}")

if __name__ == "__main__":
    main()