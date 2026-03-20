#!/usr/bin/env python3
"""
Same 3-D bumpy multimodal function as bumpy_function.py, but the CLI uses a
setter-style interface:

    python bumpy_setter.py --set x 42.0 --set y 7.5 --set z 1.3

instead of the traditional named-flag style:

    python bumpy_function.py --x 42.0 --y 7.5 --z 1.3

This interface pattern is common in frameworks that forward arbitrary
key-value overrides (e.g. Hydra, MMCV, or custom experiment runners).
Launch it with iterate2 by passing --param-setter set.
"""

import argparse
import math


def bumpy_function_3d(
    x, y, z,
    global_mu, global_sigma,
    mu_rest, sigma_rest, amps_rest,
):
    """
    3D smooth multimodal function with:
    - one global optimum = 1 at global_mu = (mx, my, mz)
    - multiple local optima < 1

    f(p) = 1 - Π_k (1 - a_k * exp(-||p - mu_k||^2 / (2 sigma_k^2)))
    """

    def sqdist(p, q):
        return (p[0] - q[0])**2 + (p[1] - q[1])**2 + (p[2] - q[2])**2

    p = (x, y, z)

    val = 1.0 - math.exp(-sqdist(p, global_mu) / (2.0 * global_sigma**2))

    for mu_k, sig_k, a_k in zip(mu_rest, sigma_rest, amps_rest):
        term = 1.0 - a_k * math.exp(-sqdist(p, mu_k) / (2.0 * sig_k**2))
        val *= term

    return 1.0 - val


def parse_args():
    parser = argparse.ArgumentParser(
        description="3-D bumpy function with setter-style CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--set",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        dest="overrides",
        default=[],
        help="Set a parameter: --set KEY VALUE (repeatable)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Convert list of [key, value] pairs to flat dict
    params = {k: v for k, v in args.overrides}

    def get_float(key, default):
        return float(params[key]) if key in params else default

    def get_int(key, default):
        return int(params[key]) if key in params else default

    x = get_float("x", 50.0)
    y = get_float("y", 50.0)
    z = get_float("z", 50.0)
    trial_number = get_int("trial_number", 0)

    global_mu_raw = params.get("global_mu", "23_42_66").split("_")
    global_mu = tuple(float(v) for v in global_mu_raw)
    global_sigma = get_float("global_sigma", 0.7)

    mu_rest_raw = params.get("mu_rest", "-2.0_0.0_0.0_2.0_0.0_0.0").split("_")
    mu_rest_flat = [float(v) for v in mu_rest_raw]
    mu_rest = [tuple(mu_rest_flat[i:i+3]) for i in range(0, len(mu_rest_flat), 3)]

    sigma_rest_raw = params.get("sigma_rest", "0.6_0.6").split("_")
    sigma_rest = [float(v) for v in sigma_rest_raw]

    amps_rest_raw = params.get("amps_rest", "0.5_0.8").split("_")
    amps_rest = [float(v) for v in amps_rest_raw]

    yval = bumpy_function_3d(
        x=x, y=y, z=z,
        global_mu=global_mu,
        global_sigma=global_sigma,
        mu_rest=mu_rest,
        sigma_rest=sigma_rest,
        amps_rest=amps_rest,
    )

    print(f'yval: {yval}, trial_number: {trial_number}')


if __name__ == "__main__":
    main()
