#!/usr/bin/env python3

import argparse
import json
import logging
import os
import subprocess
import sys
import re
import tempfile
import threading
import time
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
    parser.add_argument("--venv", default=".venv", help="Virtualenv dir (shortcut for source <venv>/bin/activate)")
    parser.add_argument(
        "--pre-run-commands",
        default=None,
        help=(
            "Shell commands to run before the training script, joined with ' && '. "
            "Useful for sourcing bashrc, activating conda/mamba envs, loading modules, etc. "
            "Example: 'source ~/.bashrc && micromamba activate gridfm'. "
            "When set, --venv is ignored."
        ),
    )
    parser.add_argument("--interpreter", default="python", help="Interpreter to use")
    parser.add_argument("--param-setter", type=str, default=None)
    parser.add_argument("--wlm", choices=["lsf", "slurm", "openshift", "vela", "none"], default="none")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--cpu-count", type=int, default=4)
    parser.add_argument("--mem-gb", type=int, default=128)
    parser.add_argument("--lsf-gpu-config-string", type=str, default=None)

    # ------------------------
    # Vela / OpenShift options
    # ------------------------
    parser.add_argument(
        "--vela-job-template",
        type=str,
        default=None,
        help="Path to the Vela job YAML template (required when --wlm vela)",
    )
    parser.add_argument(
        "--vela-chart-path",
        type=str,
        default=None,
        help="Path to the helm chart directory (required when --wlm vela)",
    )
    parser.add_argument(
        "--vela-namespace",
        type=str,
        default=None,
        help="OpenShift/Kubernetes namespace (uses current context if omitted)",
    )
    parser.add_argument(
        "--vela-cmd-placeholder",
        type=str,
        default="{{HPO_COMMAND}}",
        help="String in the job template's setupCommands that is replaced with the HPO command (default: '{{HPO_COMMAND}}')",
    )
    parser.add_argument(
        "--vela-pod-ready-timeout",
        type=int,
        default=600,
        help="Seconds to wait for the trial pod to reach Running state (default: 600)",
    )
    parser.add_argument(
        "--vela-job-timeout",
        type=int,
        default=86400,
        help="Seconds to wait for the trial job to complete (default: 86400 = 24 h)",
    )
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
    elif wlm in ("vela",):
        # Vela uses a separate submission flow; this function is not called for it.
        raise ValueError("build_launcher_command must not be called for wlm='vela'; use build_vela_job_yaml + run_vela_trial instead.")
    else:
        raise ValueError(f"Unknown WLM: {wlm}")
    logger.debug("Launcher command: %s", launcher)
    return launcher

def build_shell_command(interpreter, root_dir, script_path, venv, script_args, param_setter, underscore_to_hyphen=True, pre_run_commands=None):
    parts = [f"cd {root_dir}"]
    if pre_run_commands:
        parts.append(pre_run_commands)
        logger.debug("Pre-run commands: %s", pre_run_commands)
    elif venv:
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


def build_container_command(interpreter: str, script_path: str, script_args: dict, param_setter: Optional[str], underscore_to_hyphen: bool = True) -> str:
    """Build a bare CLI invocation suitable for running inside a container.

    Unlike :func:`build_shell_command` this function does **not** prepend
    ``cd`` or ``source venv`` – those are not needed (or available) inside an
    already-running container image.
    """
    prefix = f"{interpreter} " if interpreter else ""
    arg_list = [f"{prefix}{script_path}".strip()]
    for key, value in script_args.items():
        arg_name = key.replace("_", "-") if underscore_to_hyphen else key
        if value is None:
            logger.debug("Container cmd: skipping '%s' (None)", key)
            continue
        if param_setter:
            if isinstance(value, bool):
                if value:
                    arg_list.append(f"--{param_setter} {key}")
                else:
                    pass  # omit
            else:
                arg_list.append(f"--{param_setter} {key} {value}")
        else:
            if isinstance(value, bool):
                if value:
                    arg_list.append(f"--{arg_name}")
                # else omit
            else:
                arg_list.append(f"--{arg_name} {value}")
    cmd = " ".join(arg_list)
    logger.debug("Container command: %s", cmd)
    return cmd


def build_vela_job_yaml(
    template_path: str,
    trial_id: int,
    gpu_count: int,
    container_cmd: str,
    placeholder: str,
) -> tuple[str, str]:
    """Load *template_path* as raw text, inject HPO parameters, return ``(yaml_str, job_name)``.

    All modifications are done via targeted regex/string substitutions on the raw
    YAML text so that multi-line block scalars (e.g. awk pipelines), single-quoted
    strings, and other constructs that PyYAML would mangle on a load→dump round-trip
    are preserved exactly as written in the template.

    Changes applied:
    * ``jobName`` gets a ``-trial-{trial_id}`` suffix (unique Kubernetes resource).
    * ``numGpusPerPod`` is overwritten with *gpu_count*.
    * The *placeholder* string inside ``setupCommands`` is replaced with
      *container_cmd* in-place, preserving any surrounding wrapper (e.g. awk pipeline).
    """
    with open(template_path, "r") as fh:
        text = fh.read()

    # ── jobName ──────────────────────────────────────────────────────────────
    job_name_match = re.search(r'^(jobName\s*:\s*["\']?)([^"\'#\n]+)(["\']?)', text, re.MULTILINE)
    if not job_name_match:
        raise ValueError(f"'jobName' key not found in template '{template_path}'")
    raw_name = job_name_match.group(2).strip()
    job_name = f"{raw_name}-trial-{trial_id}"
    text = (
        text[:job_name_match.start(2)]
        + job_name_match.group(2).replace(raw_name, job_name)
        + text[job_name_match.end(2):]
    )
    logger.debug("Vela trial %d: jobName → %s", trial_id, job_name)

    # ── numGpusPerPod ────────────────────────────────────────────────────────
    text = re.sub(
        r'^(numGpusPerPod\s*:\s*)\S+',
        lambda m: f"{m.group(1)}{gpu_count}",
        text,
        flags=re.MULTILINE,
    )
    logger.debug("Vela trial %d: numGpusPerPod → %d", trial_id, gpu_count)

    # ── placeholder substitution ─────────────────────────────────────────────
    if placeholder in text:
        text = text.replace(placeholder, container_cmd)
        logger.debug("Vela trial %d: substituted placeholder '%s'", trial_id, placeholder)
    else:
        logger.warning(
            "Vela trial %d: placeholder '%s' not found in template '%s' – appending command",
            trial_id, placeholder, template_path,
        )
        text += f"\n  - {container_cmd}\n"

    return text, job_name


def _oc(*args, namespace: Optional[str] = None, check: bool = True, capture: bool = False):
    """Run an ``oc`` sub-command, optionally capturing output."""
    cmd = ["oc"] + list(args)
    if namespace:
        cmd += ["-n", namespace]
    logger.debug("oc command: %s", " ".join(cmd))
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    return subprocess.run(cmd, check=check)


def run_vela_trial(
    trial_id: int,
    job_yaml: str,
    chart_path: str,
    job_name: str,
    namespace: Optional[str],
    out_file: str,
    err_file: str,
    pod_ready_timeout: int,
    job_timeout: int,
) -> None:
    """Submit a Vela/OpenShift PyTorchJob, stream its logs, and wait for completion.

    Steps
    -----
    1. Write *job_yaml* to a temp file.
    2. ``helm template -f <tmp> <chart> | oc create [-n <ns>] -f-``
    3. Poll until the master pod (``<job_name>-master-0``) appears.
    4. ``oc logs -f <pod>`` – streams every line to stdout **and** *out_file*.
    5. After streaming ends, check the pod's terminated exit-code.
       Non-zero → raise :class:`subprocess.CalledProcessError`.
    6. Cleanup: delete the PyTorchJob resource.
    """
    ns_args = ["-n", namespace] if namespace else []
    prefix = f"[trial-{trial_id}]"

    # Write temp YAML
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix=f"vela_trial_{trial_id}_",
        delete=False,
    ) as fh:
        fh.write(job_yaml)
        tmp_yaml = fh.name
    logger.debug("Vela trial %d: temp YAML written to %s", trial_id, tmp_yaml)

    try:
        # ── 1. Submit ──────────────────────────────────────────────────────────
        ns_flag = f"-n {namespace}" if namespace else ""
        create_cmd = (
            f"helm template -f {tmp_yaml} {chart_path}"
            f" | oc create {ns_flag} -f-"
        )
        logger.info("Trial %d: submitting Vela job → %s", trial_id, create_cmd)
        result = subprocess.run(create_cmd, shell=True, capture_output=True, text=True)
        with _print_lock:
            sys.stdout.write(f"{prefix} {result.stdout}")
            sys.stdout.flush()
        if result.returncode != 0:
            raise RuntimeError(
                f"Vela trial {trial_id}: oc create failed (rc={result.returncode}):\n"
                f"{result.stderr}"
            )
        logger.info("Trial %d: job '%s' created", trial_id, job_name)

        # ── 2. Wait for master pod to appear ──────────────────────────────────
        master_pod = f"{job_name}-master-0"
        deadline = time.monotonic() + pod_ready_timeout
        logger.info("Trial %d: waiting for pod '%s' to appear (timeout %ds)…", trial_id, master_pod, pod_ready_timeout)
        while time.monotonic() < deadline:
            r = subprocess.run(
                ["oc", "get", "pod", master_pod, "--ignore-not-found"] + ns_args,
                capture_output=True, text=True,
            )
            if master_pod in r.stdout:
                logger.debug("Trial %d: pod '%s' found", trial_id, master_pod)
                break
            time.sleep(5)
        else:
            raise TimeoutError(
                f"Vela trial {trial_id}: pod '{master_pod}' did not appear within {pod_ready_timeout}s"
            )

        # ── 3. Wait for pod to be Running/Succeeded ───────────────────────────
        logger.info("Trial %d: waiting for pod '%s' to be Running…", trial_id, master_pod)
        wait_cmd = (
            ["oc", "wait", f"pod/{master_pod}",
             "--for=condition=Ready",
             f"--timeout={pod_ready_timeout}s"]
            + ns_args
        )
        wr = subprocess.run(wait_cmd, capture_output=True, text=True)
        # oc wait returns non-zero if the pod is already Completed (no Ready condition);
        # that's fine – the logs are still accessible.
        logger.debug("Trial %d: oc wait rc=%d stderr=%s", trial_id, wr.returncode, wr.stderr.strip())

        # ── 4. Stream logs ────────────────────────────────────────────────────
        log_cmd = ["oc", "logs", "-f", master_pod] + ns_args
        logger.info("Trial %d: streaming logs from '%s'", trial_id, master_pod)
        log_proc = subprocess.Popen(
            log_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        t_out = threading.Thread(
            target=_stream_pipe,
            args=(log_proc.stdout, out_file, trial_id, "stdout", sys.stdout),
            daemon=True,
        )
        t_err = threading.Thread(
            target=_stream_pipe,
            args=(log_proc.stderr, err_file, trial_id, "stderr", sys.stderr),
            daemon=True,
        )
        t_out.start()
        t_err.start()
        log_proc.wait(timeout=job_timeout)
        t_out.join()
        t_err.join()
        logger.debug("Trial %d: log stream ended (rc=%d)", trial_id, log_proc.returncode)

        # ── 4b. If oc logs exited early (e.g. "unexpected EOF"), the pod may
        #        still be running.  Re-attach the log stream and wait for it to
        #        finish so we capture the full output and don't delete a live job.
        if log_proc.returncode != 0:
            logger.warning(
                "Trial %d: oc logs exited with rc=%d (possible EOF disconnect) – "
                "waiting for pod to terminate before reading exit code",
                trial_id, log_proc.returncode,
            )
            # Wait for pod phase Succeeded or Failed (container terminated).
            oc_wait_phase = subprocess.run(
                ["oc", "wait", f"pod/{master_pod}",
                 "--for=jsonpath={.status.phase}=Succeeded",
                 f"--timeout={job_timeout}s"]
                + ns_args,
                capture_output=True, text=True,
            )
            if oc_wait_phase.returncode != 0:
                # Pod may have Failed; try that phase too.
                subprocess.run(
                    ["oc", "wait", f"pod/{master_pod}",
                     "--for=jsonpath={.status.phase}=Failed",
                     f"--timeout=30s"]
                    + ns_args,
                    capture_output=True, text=True,
                )
            # Re-stream any log lines written after the disconnect into the same files.
            catchup = subprocess.run(
                ["oc", "logs", "--tail=-1", master_pod] + ns_args,
                capture_output=True, text=True,
            )
            if catchup.stdout:
                with open(out_file, "a", encoding="utf-8", errors="replace") as fh:
                    fh.write(catchup.stdout)
            if catchup.stderr:
                with open(err_file, "a", encoding="utf-8", errors="replace") as fh:
                    fh.write(catchup.stderr)

        # ── 5. Check pod exit code ────────────────────────────────────────────
        # Poll until the pod has a terminated exit code (handles the race
        # between oc-logs EOF and pod termination being recorded in the API).
        exit_code_str = ""
        for _attempt in range(30):
            ec_result = subprocess.run(
                ["oc", "get", "pod", master_pod, "-o",
                 "jsonpath={.status.containerStatuses[0].state.terminated.exitCode}"]
                + ns_args,
                capture_output=True, text=True,
            )
            exit_code_str = ec_result.stdout.strip()
            if exit_code_str.lstrip("-").isdigit():
                break
            logger.debug("Trial %d: exit code not yet available, retrying in 5 s…", trial_id)
            time.sleep(5)
        exit_code = int(exit_code_str) if exit_code_str.lstrip("-").isdigit() else 0
        logger.info("Trial %d: pod exit code = %s", trial_id, exit_code)
        if exit_code != 0:
            logger.warning("Trial %d: pod exited with code %d – marking trial as pruned", trial_id, exit_code)
            raise optuna.exceptions.TrialPruned(f"pod exited with code {exit_code}")

    finally:
        # ── 6. Cleanup – delete the job ───────────────────────────────────────
        logger.debug("Trial %d: deleting PyTorchJob '%s'", trial_id, job_name)
        subprocess.run(
            ["oc", "delete", "pytorchjob", job_name, "--ignore-not-found"] + ns_args,
            capture_output=True,
        )
        try:
            os.unlink(tmp_yaml)
        except OSError:
            pass

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
    """Extract metric values from a log file.

    Each entry in *metric_names* is either a plain name (uses the **last**
    match) or ``name#N`` to select the **N-th occurrence** (0-based). This
    lets you disambiguate scripts that print the same metric key multiple
    times, e.g.::

        metrics:
          - "Samples/sec#0"   # DataLoader throughput (first occurrence)
          - "Samples/sec#1"   # Training throughput   (second occurrence)
          - "Samples/sec#2"   # Inference throughput  (third occurrence)
          - GFLOPS
    """
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
        # Support  name#N  syntax for Nth-occurrence selection (0-based)
        occurrence: Optional[int] = None
        bare_metric = metric
        idx_match = re.fullmatch(r'(.+)#(\d+)', metric)
        if idx_match:
            bare_metric = idx_match.group(1)
            occurrence = int(idx_match.group(2))

        # Matches: key: value | key=value | [performance] key : value | Lightning table │ key │ value │
        pattern = re.compile(
            rf"(?:\[\w+\]\s*)?{re.escape(bare_metric)}\s*(?:[:=│])\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        )
        matches = pattern.findall(text)
        if not matches:
            logger.warning("Metric '%s' not found in '%s' — defaulting to 0.0", metric, path)
            results.append(0.0)
        elif occurrence is not None:
            if occurrence >= len(matches):
                logger.warning(
                    "Metric '%s' occurrence #%d requested but only %d match(es) found — defaulting to 0.0",
                    metric, occurrence, len(matches),
                )
                results.append(0.0)
            else:
                value = float(matches[occurrence])
                logger.debug("Metric '%s': using occurrence #%d = %s", metric, occurrence, value)
                results.append(value)
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

        if args.wlm == "vela":
            # ── Vela / OpenShift path ──────────────────────────────────────
            if not args.vela_job_template:
                raise ValueError("--vela-job-template is required when --wlm vela")
            if not args.vela_chart_path:
                raise ValueError("--vela-chart-path is required when --wlm vela")
            container_cmd = build_container_command(
                args.interpreter, script_path, script_args,
                args.param_setter, args.underscore_to_hyphen,
            )
            logger.info("Trial %d: container command → %s", trial.number, container_cmd)
            job_yaml, job_name = build_vela_job_yaml(
                args.vela_job_template,
                trial.number,
                gpu_count,
                container_cmd,
                args.vela_cmd_placeholder,
            )
            logger.debug("Trial %d: job YAML (first 400 chars):\n%s", trial.number, job_yaml[:400])
            run_vela_trial(
                trial_id=trial.number,
                job_yaml=job_yaml,
                chart_path=args.vela_chart_path,
                job_name=job_name,
                namespace=args.vela_namespace,
                out_file=out_file,
                err_file=err_file,
                pod_ready_timeout=args.vela_pod_ready_timeout,
                job_timeout=args.vela_job_timeout,
            )
        else:
            # ── Standard WLM path (lsf / slurm / none) ────────────────────
            shell_cmd = build_shell_command(args.interpreter, root_dir, script_path, args.venv, script_args, args.param_setter, args.underscore_to_hyphen, pre_run_commands=args.pre_run_commands)
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
        journal_path = args.optuna_db_path[len("js:///"):]
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
    study.optimize(
        objective,
        n_trials=args.optuna_n_trials,
        n_jobs=args.parallelism,
        catch=(Exception,),   # mark trial as FAILED and continue; never crash the study
    )

    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("Pareto Front Trials: %d", len(study.best_trials))
    for t in study.best_trials:
        logger.info("  Trial %d: Values=%s  Params=%s", t.number, t.values, t.params)

if __name__ == "__main__":
    main()