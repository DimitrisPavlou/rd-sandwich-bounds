#!/usr/bin/env python
"""CLI for the R-D lower bound, torch port of ``rdlb.py``.

Flat-flag CLI (like train_rdub.py) so it can be driven by a YAML config via
``scripts/run_sweep.py``. ``--command train`` (default) trains the log-u model
and, with ``--eval_after``, immediately runs the exhaustive-optimizer eval and
writes an ``rd-*.npz`` (the number the plot consumes). ``--command eval``
re-evaluates an existing checkpoint (requires ``--ckpt`` and ``--y_init exhaustive``).

Train + auto-eval (matches README.md's LB command; one config combo):

    python train/train_rdlb.py --dataset gaussian --data_dim 1000 \
        --model mlp --units 100000,100000,100000 --lamb 100 \
        --checkpoint_dir checkpoints/gaussian \
        --command train --batchsize 1024 --num_Ck_samples 2 --last_step 3000 \
        --y_init quick --y_quick_topn 10 --lr 5e-4 --eval_after -V

Or the whole sweep from a config:  python scripts/run_sweep.py --config configs/gaussian_lb.yaml
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdsandwich.dataloader import build_loader, get_source
from rdsandwich.lower_bound import (
    LowerBoundTrainer, RDLBTrainConfig, build_log_u_model, estimate_R_lower_bound,
)
from rdsandwich.utils import (
    JsonlLogger, config_dict_to_str, get_device, get_time_str, load_checkpoint, seed_everything,
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--verbose", "-V", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--chunksize", type=int, default=None)
    p.add_argument("--device", default=None)

    p.add_argument("--data_dim", type=int, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--gparams_path", default=None)
    p.add_argument("--in_channels", type=int, default=3)

    p.add_argument("--lamb", type=float, required=True)
    p.add_argument("--model", default="mlp")
    p.add_argument("--units", type=lambda s: [int(i) for i in s.split(",") if i], default=[])
    p.add_argument("--hidden_units_per_dim", type=int, default=0,
                   help="If >0 and --units is empty, build --num_hidden_layers hidden layers of "
                        "(hidden_units_per_dim * data_dim) units. Lets a data_dim sweep auto-size the "
                        "net; the paper (A.5.2) uses 20 (i.e. 20n units) for the Gaussian LB.")
    p.add_argument("--num_hidden_layers", type=int, default=2)
    p.add_argument("--activation", default="selu")
    p.add_argument("--kernel_dims", type=lambda s: [int(i) for i in s.split(",") if i], default=[])

    p.add_argument("--command", choices=["train", "eval"], default="train")

    # Shared C_k / optimize_y knobs (train and eval).
    p.add_argument("--batchsize", "-k", type=int, default=1024)
    p.add_argument("--num_Ck_samples", "-M", type=int, default=1)
    p.add_argument("--y_init", default="quick", choices=["quick", "exhaustive"])
    p.add_argument("--y_quick_topn", type=int, default=10)
    p.add_argument("--y_steps", type=int, default=500)
    p.add_argument("--y_tol", type=float, default=1e-6)
    p.add_argument("--y_lr", type=float, default=1e-2)

    # Train-only.
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--last_step", type=int, default=3000)
    p.add_argument("--checkpoint_interval", type=int, default=1000)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--log_E_Ck_max_delta", type=float, default=1.0)
    p.add_argument("--eval_after", action="store_true",
                   help="After training, run the exhaustive-optimizer eval and save rd-*.npz.")
    p.add_argument("--eval_num_Ck_samples", type=int, default=5,
                   help="M for the post-training eval (--eval_after).")

    # Eval-only.
    p.add_argument("--ckpt", default=None, help="Checkpoint to evaluate (required for --command eval).")

    return p.parse_args()


def get_runname(args):
    return config_dict_to_str(
        vars(args), record_keys=("data_dim", "model", "units", "lamb", "batchsize", "seed"),
        prefix="rdlb",
    )


def run_eval_and_save(model, source, args, save_dir, device, num_Ck_samples, y_init):
    """Run est_R_ with the (exhaustive) global optimizer and save rd-*.npz."""
    import numpy as np

    cfg = RDLBTrainConfig(
        lamb=args.lamb, batchsize=args.batchsize, num_Ck_samples=num_Ck_samples,
        y_steps=args.y_steps, y_lr=args.y_lr, y_tol=args.y_tol,
        y_init=y_init, y_quick_topn=args.y_quick_topn, chunksize=args.chunksize,
    )
    res = estimate_R_lower_bound(model, source, args.lamb, cfg, device=device)
    print(f"R_ = {res['R_']:.4f} nats/sample (linear lower-bound intercept at slope -{args.lamb})")
    save_path = os.path.join(
        save_dir, f"rd-seed={args.seed}-k={args.batchsize}-M={num_Ck_samples}-lamb={args.lamb:.5g}-R_={res['R_']:.3f}.npz"
    )
    np.savez(save_path, **res)
    print(f"Saved final R_ estimate to {save_path}")


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)

    if not args.units and args.hidden_units_per_dim > 0:
        # auto-size the net from data_dim (e.g. paper A.5.2's 20n), so a data_dim sweep works
        args.units = [args.hidden_units_per_dim * args.data_dim] * args.num_hidden_layers
    model = build_log_u_model(
        args.model, args.data_dim, args.units,
        activation=args.activation, in_channels=args.in_channels, kernel_dims=args.kernel_dims,
    ).to(device)
    source = get_source(args.dataset, data_dim=args.data_dim, gparams_path=args.gparams_path, seed=args.seed, device=device)

    save_dir = os.path.join(args.checkpoint_dir, get_runname(args))
    os.makedirs(save_dir, exist_ok=True)

    if args.command == "train":
        cfg = RDLBTrainConfig(
            lamb=args.lamb, batchsize=args.batchsize, num_Ck_samples=args.num_Ck_samples,
            last_step=args.last_step, lr=args.lr, y_steps=args.y_steps, y_lr=args.y_lr, y_tol=args.y_tol,
            y_init=args.y_init, y_quick_topn=args.y_quick_topn, beta=args.beta,
            log_E_Ck_max_delta=args.log_E_Ck_max_delta, chunksize=args.chunksize,
            checkpoint_interval=args.checkpoint_interval,
        )
        log_path = os.path.join(save_dir, f"record-{get_time_str()}.jsonl")
        print(f"Logging to {log_path}")

        loader = build_loader(source, cfg.batchsize)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        trainer = LowerBoundTrainer(
            model, loader, optimizer=optimizer, cfg=cfg, device=device,
            logger=JsonlLogger(log_path), ckpt_dir=save_dir, verbose=args.verbose,
        )
        trainer.train()

        if args.eval_after:
            print("Post-training eval with the exhaustive global optimizer...")
            run_eval_and_save(model, source, args, save_dir, device,
                              num_Ck_samples=args.eval_num_Ck_samples, y_init="exhaustive")

    elif args.command == "eval":
        assert args.ckpt, "--command eval requires --ckpt"
        assert args.y_init == "exhaustive", "Should run full global optimization to be correct."
        load_checkpoint(args.ckpt, model, map_location=device)
        print(f"Loaded model weights from {args.ckpt}")
        run_eval_and_save(model, source, args, save_dir, device,
                          num_Ck_samples=args.num_Ck_samples, y_init=args.y_init)


if __name__ == "__main__":
    main()
