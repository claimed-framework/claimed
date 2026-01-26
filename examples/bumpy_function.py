#!/usr/bin/env python3
import argparse
import math


def bumpy_function_3d(
    x, y, z,
    global_mu, global_sigma,
    mu_rest, sigma_rest, amps_rest,
):
    """
    3D smooth multimodal function with:
    - one global optimum = 1 at global_mu = (mx,my,mz)
    - multiple local optima < 1

    f(p) = 1 - Π_k (1 - a_k * exp(-||p - mu_k||^2 / (2 sigma_k^2)))
    """

    def sqdist(p, q):
        return (p[0] - q[0])**2 + (p[1] - q[1])**2 + (p[2] - q[2])**2

    p = (x, y, z)

    # Global peak (amplitude = 1)
    val = 1.0 - math.exp(
        -sqdist(p, global_mu) / (2.0 * global_sigma**2)
    )

    # Local peaks
    for mu_k, sig_k, a_k in zip(mu_rest, sigma_rest, amps_rest):
        term = 1.0 - a_k * math.exp(
            -sqdist(p, mu_k) / (2.0 * sig_k**2)
        )
        val *= term

    return 1.0 - val


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate the 3D bumpy multimodal function.")

    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--trial-number", type=int, default=0)

    parser.add_argument(
        "--global-mu",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("MX", "MY", "MZ"),
    )
    parser.add_argument("--global-sigma", type=float, default=0.7)

    parser.add_argument(
        "--mu-rest",
        type=float,
        nargs="*",
        default=[-2.0, 0.0, 0.0,  2.0, 0.0, 0.0],
        help="Flat list of (x y z) triplets",
    )
    parser.add_argument(
        "--sigma-rest",
        type=float,
        nargs="*",
        default=[0.6, 0.6],
    )
    parser.add_argument(
        "--amps-rest",
        type=float,
        nargs="*",
        default=[0.5, 0.8],
    )

    args = parser.parse_args()

    mu_rest = [
        tuple(args.mu_rest[i:i+3])
        for i in range(0, len(args.mu_rest), 3)
    ]

    yval = bumpy_function_3d(
        x=args.x,
        y=args.y,
        z=args.z,
        global_mu=tuple(args.global_mu),
        global_sigma=args.global_sigma,
        mu_rest=mu_rest,
        sigma_rest=args.sigma_rest,
        amps_rest=args.amps_rest,
    )

    print(f'yval: {yval}, trial_number: {args.trial_number}')
