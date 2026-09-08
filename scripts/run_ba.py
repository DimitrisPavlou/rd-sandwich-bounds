#!/usr/bin/env python
"""CLI for the Blahut-Arimoto reference algorithm, port of ``ba.py``.

Example:

    python scripts/run_ba.py --samples data/banana-dim=2-samples=100000.npy \
        --lamb 1.0 --bins 80 --steps 2000 --tol 1e-5 --save_dir checkpoints/banana/BA
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdsandwich.utils.ba import discretize_and_run_ba
from rdsandwich.utils import MyJSONEncoder, config_dict_to_str


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=str, required=True, help="Path to .npy file of data samples.")
    p.add_argument("--lamb", type=float, default=1.0)
    p.add_argument("--save_dir", default="./results/")
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("-V", "--verbose", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed)
    samples = np.load(args.samples)
    records, log_Q, log_q_y = discretize_and_run_ba(
        samples, lamb=args.lamb, bins=args.bins, steps=args.steps, tol=args.tol, verbose=args.verbose
    )

    final = records[-1]
    runname = config_dict_to_str(vars(args), record_keys=("lamb", "bins"), prefix="ba", use_abbr=False)
    save_dir = os.path.join(args.save_dir, runname)
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"steps={final['step'] + 1}-R={final['R']:.3g}-D={final['D']:.3g}-obj={final['ub']:.3g}.jsonl"
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, "a") as f:
        for rcd in records:
            json.dump(rcd, f, cls=MyJSONEncoder)
            f.write("\n")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
