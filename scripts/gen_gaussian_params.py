#!/usr/bin/env python
"""Port of ``gen_gaussian_params.py``: sample and save a random factorized
Gaussian source's (loc, scale), for reproducible n-dimensional experiments.

    python scripts/gen_gaussian_params.py --save_dir data --dim 1000
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdsandwich.dataloader import gen_gaussian_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", default="./data")
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    params = gen_gaussian_params(args.dim, seed=args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"gaussian_params-dim={args.dim}.npz")
    np.savez(path, **params)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
