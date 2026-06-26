#!/usr/bin/env python3
"""
iterate2 WLM plugin – Vela / OpenShift PyTorchJob (MLBatch)

Submits each HPO trial as a PyTorchJob on an OpenShift cluster via
``helm template | oc create``, streams pod logs, checks the exit code,
and cleans up the job resource.

Environment variables provided by iterate2
------------------------------------------
  ITERATE_TRIAL_NUMBER          integer trial ID
  ITERATE_TRIAL_CONTAINER_CMD   bare CLI invocation for inside the container
                                (no ``cd``, no ``source venv`` – use this
                                 one, not ITERATE_TRIAL_CMD)
  ITERATE_OUT_FILE              file to write trial stdout
  ITERATE_ERR_FILE              file to write trial stderr

WLM configuration (from the ``wlm:`` section in the HPO YAML)
-------------------------------------------------------------
All keys from ``wlm:`` are available as ``ITERATE_WLM_<KEY_UPPERCASE>``
(hyphens → underscores).  Recognised keys:

  job-template        (ITERATE_WLM_JOB_TEMPLATE)         REQUIRED path to PyTorchJob helm values YAML
  chart-path          (ITERATE_WLM_CHART_PATH)           REQUIRED path to pytorchjob-generator helm chart
  namespace           (ITERATE_WLM_NAMESPACE)            optional; uses current oc context if omitted
  cmd-placeholder     (ITERATE_WLM_CMD_PLACEHOLDER)      default: {{HPO_COMMAND}}
  gpu-count           (ITERATE_WLM_GPU_COUNT)            default: 1
  pod-ready-timeout   (ITERATE_WLM_POD_READY_TIMEOUT)    seconds; default: 600
  job-timeout         (ITERATE_WLM_JOB_TIMEOUT)          seconds; default: 86400

Usage in HPO YAML
-----------------
  wlm:
    job-template:   examples/vela_gridfm_template.yaml
    chart-path:     ~/tmp/mlbatch/tools/pytorchjob-generator/chart
    namespace:      my-project
    cmd-placeholder: "{{HPO_COMMAND}}"
    gpu-count:      1
    pod-ready-timeout: 600
    job-timeout:    86400

Usage in the launch script
--------------------------
  iterate2 \\
    --wlm-plugin "$(dirname "$0")/wlm_plugins/vela_plugin.py" \\
    --hpo-yaml   my_hpo.yaml \\
    --no-underscore-to-hyphen \\
    ...

Exit code
---------
Exits 0 on success, 1 on failure.  iterate2 marks the Optuna trial as
FAILED on any non-zero exit.
"""

import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional


# ── helpers ───────────────────────────────────────────────────────────────────

def env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.environ.get(key, default)
    if required and not val:
        sys.exit(f"[vela_plugin] ERROR: required env var '{key}' is not set")
    return val or ""


def patch_job_yaml(template_path: str, trial_id: int, gpu_count: int,
                   container_cmd: str, placeholder: str) -> tuple[str, str]:
    """Patch the helm values YAML; return (patched_text, job_name)."""
    with open(template_path) as fh:
        text = fh.read()

    # jobName → append -trial-<N>
    m = re.search(r'^(jobName\s*:\s*["\']?)([^"\'#\n]+)(["\']?)', text, re.MULTILINE)
    if not m:
        sys.exit(f"[vela_plugin] ERROR: 'jobName' key not found in '{template_path}'")
    raw_name = m.group(2).strip()
    job_name = f"{raw_name}-trial-{trial_id}"
    text = text[:m.start(2)] + m.group(2).replace(raw_name, job_name) + text[m.end(2):]

    # numGpusPerPod → overwrite
    text = re.sub(
        r'^(numGpusPerPod\s*:\s*)\S+',
        lambda m2: f"{m2.group(1)}{gpu_count}",
        text, flags=re.MULTILINE,
    )

    # placeholder → container_cmd
    if placeholder in text:
        text = text.replace(placeholder, container_cmd)
    else:
        print(f"[vela_plugin] WARNING: placeholder '{placeholder}' not found in template – appending")
        text += f"\n  - {container_cmd}\n"

    return text, job_name


def stream_pipe(pipe, dest_file: str, prefix: str, dest_stream):
    with open(dest_file, "w", encoding="utf-8", errors="replace") as fh:
        for raw in pipe:
            line = raw.decode("utf-8", errors="replace")
            fh.write(line)
            fh.flush()
            dest_stream.write(f"{prefix} {line}")
            dest_stream.flush()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    trial_id   = int(env("ITERATE_TRIAL_NUMBER", required=True))
    cmd        = env("ITERATE_TRIAL_CONTAINER_CMD", required=True)
    out_file   = env("ITERATE_OUT_FILE", required=True)
    err_file   = env("ITERATE_ERR_FILE", required=True)

    template   = env("ITERATE_WLM_JOB_TEMPLATE",     required=True)
    chart      = env("ITERATE_WLM_CHART_PATH",        required=True)
    namespace  = env("ITERATE_WLM_NAMESPACE",         "")
    placeholder = env("ITERATE_WLM_CMD_PLACEHOLDER",  "{{HPO_COMMAND}}")
    gpu_count  = int(env("ITERATE_WLM_GPU_COUNT",     "1"))
    pod_timeout = int(env("ITERATE_WLM_POD_READY_TIMEOUT", "600"))
    job_timeout = int(env("ITERATE_WLM_JOB_TIMEOUT",       "86400"))

    ns_args = ["-n", namespace] if namespace else []
    prefix  = f"[trial-{trial_id}]"

    # Resolve ~ in paths
    template = str(Path(template).expanduser())
    chart    = str(Path(chart).expanduser())

    print(f"{prefix} Patching template {template}")
    job_yaml, job_name = patch_job_yaml(template, trial_id, gpu_count, cmd, placeholder)
    print(f"{prefix} Job name: {job_name}")

    # Write patched values to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix=f"vela_trial_{trial_id}_", delete=False
    ) as fh:
        fh.write(job_yaml)
        tmp_yaml = fh.name

    try:
        # ── Submit ────────────────────────────────────────────────────────────
        ns_flag = f"-n {namespace}" if namespace else ""
        create_cmd = f"helm template -f {tmp_yaml} {chart} | oc create {ns_flag} -f-"
        print(f"{prefix} Submitting: {create_cmd}")
        result = subprocess.run(create_cmd, shell=True, capture_output=True, text=True)
        sys.stdout.write(result.stdout)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            sys.exit(f"{prefix} ERROR: oc create failed (rc={result.returncode})")

        master_pod = f"{job_name}-master-0"

        # ── Wait for pod to appear ────────────────────────────────────────────
        deadline = time.monotonic() + pod_timeout
        print(f"{prefix} Waiting for pod {master_pod} …")
        while time.monotonic() < deadline:
            r = subprocess.run(
                ["oc", "get", "pod", master_pod, "--ignore-not-found"] + ns_args,
                capture_output=True, text=True,
            )
            if master_pod in r.stdout:
                break
            time.sleep(5)
        else:
            sys.exit(f"{prefix} ERROR: pod '{master_pod}' did not appear within {pod_timeout}s")

        # ── Wait for pod Ready (best-effort) ─────────────────────────────────
        subprocess.run(
            ["oc", "wait", f"pod/{master_pod}", "--for=condition=Ready",
             f"--timeout={pod_timeout}s"] + ns_args,
            capture_output=True, text=True,
        )

        # ── Stream logs ───────────────────────────────────────────────────────
        print(f"{prefix} Streaming logs from {master_pod}")
        log_proc = subprocess.Popen(
            ["oc", "logs", "-f", master_pod] + ns_args,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        t_out = threading.Thread(
            target=stream_pipe,
            args=(log_proc.stdout, out_file, prefix, sys.stdout), daemon=True,
        )
        t_err = threading.Thread(
            target=stream_pipe,
            args=(log_proc.stderr, err_file, prefix, sys.stderr), daemon=True,
        )
        t_out.start(); t_err.start()
        try:
            log_proc.wait(timeout=job_timeout)
        except subprocess.TimeoutExpired:
            log_proc.kill()
        t_out.join(); t_err.join()

        # Catch-up read in case of early EOF disconnect
        if log_proc.returncode != 0:
            catchup = subprocess.run(
                ["oc", "logs", "--tail=-1", master_pod] + ns_args,
                capture_output=True, text=True,
            )
            if catchup.stdout:
                with open(out_file, "a") as fh:
                    fh.write(catchup.stdout)
            if catchup.stderr:
                with open(err_file, "a") as fh:
                    fh.write(catchup.stderr)

        # ── Check exit code ───────────────────────────────────────────────────
        exit_code_str = ""
        for _ in range(30):
            ec = subprocess.run(
                ["oc", "get", "pod", master_pod, "-o",
                 "jsonpath={.status.containerStatuses[0].state.terminated.exitCode}"]
                + ns_args,
                capture_output=True, text=True,
            )
            exit_code_str = ec.stdout.strip()
            if exit_code_str.lstrip("-").isdigit():
                break
            time.sleep(5)

        exit_code = int(exit_code_str) if exit_code_str.lstrip("-").isdigit() else 0
        print(f"{prefix} Pod exit code: {exit_code}")
        if exit_code != 0:
            sys.exit(f"{prefix} Trial FAILED: pod exited with code {exit_code}")

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────────
        subprocess.run(
            ["oc", "delete", "pytorchjob", job_name, "--ignore-not-found"] + ns_args,
            capture_output=True,
        )
        try:
            os.unlink(tmp_yaml)
        except OSError:
            pass


if __name__ == "__main__":
    main()
