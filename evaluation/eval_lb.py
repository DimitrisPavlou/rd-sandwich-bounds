#!/usr/bin/env python
"""Estimate R_L(D) for a trained R-D LOWER-BOUND model (``est_R_``).

Pass the same flags as for ``train/train_lb.py``: they determine the run
directory, whose newest checkpoint is loaded (or ``--ckpt``). Runs the
exhaustive global inner optimizer and writes
``<run dir>/rd-seed=...-k=...-M=...-lamb=...-R_=....npz`` (read by
``evaluation/plot_rdlb.py`` and ``evaluation/plot_rdub.py``).

    python evaluation/eval_lb.py --dataset gaussian --data_dim 1000 \
        --model mlp --units 100000,100000,100000 --lamb 100 \
        --checkpoint_dir checkpoints/gaussian --num_Ck_samples 5
"""
import os

import numpy as np

from rdsandwich.cli.common import load_dataset
from rdsandwich.cli.lower_bound import (
    build_eval_parser,
    build_model,
    ck_config,
    finalize_args,
    run_dir,
)
from rdsandwich.lower_bound.evaluate import estimate_R_lower_bound
from rdsandwich.utils.torch_utils import get_device, seed_everything
from rdsandwich.utils.io import latest_checkpoint, load_checkpoint


def main():
    args = build_eval_parser().parse_args()
    if args.y_init != "exhaustive":
        raise SystemExit("eval_lb needs --y_init exhaustive: the bound is only valid with the "
                         "full global optimization.")
    seed_everything(args.seed)
    device = get_device(args.device)

    dataset = load_dataset(args, device)
    finalize_args(args, dataset)
    model = build_model(args).to(device)

    save_dir = run_dir(args)
    ckpt = args.ckpt or latest_checkpoint(save_dir)
    if ckpt is None:
        raise SystemExit(f"No checkpoint found in {save_dir!r}; train this run first.")
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()
    print(f"Loaded {ckpt}")

    cfg = ck_config(args)
    res = estimate_R_lower_bound(model, dataset, args.lamb, cfg, device=device)
    print(f"R_ = {res['R_']:.4f} nats/sample (linear lower-bound intercept at slope -{args.lamb})")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"rd-seed={args.seed}-k={args.batchsize}-M={args.num_Ck_samples}-lamb={args.lamb:.5g}-R_={res['R_']:.3f}.npz",
    )
    np.savez(save_path, **res)
    print(f"Saved final R_ estimate to {save_path}")


if __name__ == "__main__":
    main()
