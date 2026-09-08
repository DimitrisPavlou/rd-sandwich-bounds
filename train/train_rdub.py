#!/usr/bin/env python
"""CLI for the R-D upper bound (beta-VAE), torch port of ``rdub_mlp.py``.

Builds an ``RDUBModel`` + a DataLoader for the chosen source and trains it with
``UpperBoundTrainer``. Example (n=1000 Gaussian, one lambda/latent_dim combo;
see experiments/gaussian.sh or configs/gaussian_ub.yaml for the full sweep):

    python train/train_rdub.py \
        --dataset gaussian --gparams_path data/gaussian/gaussian_params-dim=1000.npz \
        --data_dim 1000 --latent_dim 1000 --prior_type gmm_1 \
        --decoder_units 0 --lambda 10 --checkpoint_dir checkpoints/gaussian \
        --epochs 80 --steps_per_epoch 1000 --lr 5e-4 --batchsize 64
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdsandwich.dataloader import build_loader, get_source
from rdsandwich.upper_bound import (
    RDUBConfig, RDUBModel, UpperBoundTrainer, check_no_decoder, make_lr_scheduler,
)
from rdsandwich.utils import (
    JsonlLogger, config_dict_to_str, get_device, get_time_str, seed_everything,
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--device", default=None)

    p.add_argument("--data_dim", type=int, required=True)
    p.add_argument("--dataset", type=str, default="gaussian")
    p.add_argument("--gparams_path", type=str, default=None)

    p.add_argument("--encoder_units", type=lambda s: [int(i) for i in s.split(",") if i], default=[])
    p.add_argument("--decoder_units", type=lambda s: [int(i) for i in s.split(",") if i], default=[])
    p.add_argument("--encoder_activation", default="softplus")
    p.add_argument("--decoder_activation", default="softplus")
    p.add_argument("--latent_dim", type=int, default=None)
    p.add_argument("--posterior_type", default="gaussian")
    p.add_argument("--prior_type", default="std_gaussian")
    p.add_argument("--ar_hidden_units", type=lambda s: [int(i) for i in s.split(",") if i], default=[10, 10])
    p.add_argument("--maf_stacks", type=int, default=0)
    p.add_argument("--lambda", type=float, default=0.01, dest="lmbda")
    p.add_argument("--rpd", action="store_true")
    p.add_argument("--nats", action="store_true")

    p.add_argument("--batchsize", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps_per_epoch", type=int, default=1000)
    p.add_argument("--verbose", "-V", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)

    if check_no_decoder(args.decoder_units):
        print(f"Using Z=Y; resetting latent_dim={args.latent_dim} to data_dim={args.data_dim}")
        args.latent_dim = args.data_dim

    cfg = RDUBConfig(
        data_dim=args.data_dim, latent_dim=args.latent_dim, lmbda=args.lmbda,
        encoder_units=args.encoder_units, decoder_units=args.decoder_units,
        encoder_activation=args.encoder_activation, decoder_activation=args.decoder_activation,
        prior_type=args.prior_type, posterior_type=args.posterior_type,
        ar_hidden_units=args.ar_hidden_units, maf_stacks=args.maf_stacks,
        rpd=args.rpd, nats=args.nats,
    )
    model = RDUBModel(cfg).to(device)
    source = get_source(args.dataset, data_dim=args.data_dim, gparams_path=args.gparams_path, seed=args.seed, device=device)
    loader = build_loader(source, args.batchsize)

    runname = config_dict_to_str(
        vars(args),
        record_keys=("data_dim", "latent_dim", "lmbda", "encoder_units", "decoder_units",
                     "prior_type", "posterior_type", "maf_stacks"),
        prefix="rdub",
    )
    save_dir = os.path.join(args.checkpoint_dir, runname)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"record-{get_time_str()}.jsonl")
    ckpt_path = os.path.join(save_dir, f"ckpt-lmbda={args.lmbda}-epoch={args.epochs}.pt")
    print(f"Logging to {log_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_lr_scheduler(optimizer, args.epochs)
    trainer = UpperBoundTrainer(
        model, loader, optimizer=optimizer, epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch, device=device, scheduler=scheduler,
        logger=JsonlLogger(log_path), ckpt_path=ckpt_path, verbose=args.verbose,
    )
    trainer.train()
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
