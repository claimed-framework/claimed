#!/usr/bin/env python3
"""
Bumpy 3-D multimodal function — called by iterate2 as a trial script.

iterate2 sets the following environment variables before calling this script:
  ITERATE_TRIAL_NUMBER   – integer trial index
  ITERATE_OUT_FILE       – path where metrics must be written
  ITERATE_ERR_FILE       – path for error logging
  ITERATE_PARAM_X        – HPO parameter x
  ITERATE_PARAM_Y        – HPO parameter y
  ITERATE_PARAM_Z        – HPO parameter z
  ITERATE_PARAM_GLOBAL_MU – static parameter (three space-separated floats)

All output that iterate2 uses to extract metrics must be written to
ITERATE_OUT_FILE (not stdout), one metric per line in "name: value" format.
"""

import math
import os
import sys


def bumpy_function_3d(
    x, y, z,
    global_mu, global_sigma,
    mu_rest, sigma_rest, amps_rest,
):
    """
    3D smooth multimodal function.
      - one global optimum = 1 at global_mu = (mx, my, mz)
      - multiple local optima < 1

    f(p) = 1 - prod_k (1 - a_k * exp(-||p - mu_k||^2 / (2 sigma_k^2)))
    """

    def sqdist(p, q):
        return (p[0] - q[0])**2 + (p[1] - q[1])**2 + (p[2] - q[2])**2

    p = (x, y, z)

    val = 1.0 - math.exp(-sqdist(p, global_mu) / (2.0 * global_sigma**2))

    for mu_k, sig_k, a_k in zip(mu_rest, sigma_rest, amps_rest):
        val *= 1.0 - a_k * math.exp(-sqdist(p, mu_k) / (2.0 * sig_k**2))

    return 1.0 - val


if __name__ == "__main__":
    # --- read parameters from environment ---------------------------------- #
    try:
        x = float(os.environ["ITERATE_PARAM_X"])
        y = float(os.environ["ITERATE_PARAM_Y"])
        z = float(os.environ["ITERATE_PARAM_Z"])
        global_mu = tuple(map(float, os.environ["ITERATE_PARAM_GLOBAL_MU"].split()))
        out_file  = os.environ["ITERATE_OUT_FILE"]
        trial_num = os.environ.get("ITERATE_TRIAL_NUMBER", "?")
    except KeyError as exc:
        print(f"ERROR: missing required environment variable {exc}", file=sys.stderr)
        sys.exit(1)

    if len(global_mu) != 3:
        print("ERROR: ITERATE_PARAM_GLOBAL_MU must contain exactly three floats", file=sys.stderr)
        sys.exit(1)

    # Fixed defaults for the local-optima configuration
    mu_rest    = [(-2.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    sigma_rest = [0.6, 0.6]
    amps_rest  = [0.5, 0.8]
    global_sigma = 0.7

    # --- evaluate ---------------------------------------------------------- #
    yval = bumpy_function_3d(
        x=x, y=y, z=z,
        global_mu=global_mu,
        global_sigma=global_sigma,
        mu_rest=mu_rest,
        sigma_rest=sigma_rest,
        amps_rest=amps_rest,
    )

    # --- write metrics to ITERATE_OUT_FILE --------------------------------- #
    with open(out_file, "w") as fh:
        fh.write(f"yval: {yval}\n")

    print(f"[trial-{trial_num}] yval={yval:.6f}")
