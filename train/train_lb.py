#!/usr/bin/env python
"""Train an R-D LOWER-BOUND ``log u`` model (Algorithm 1).

``--dataset`` is a synthetic source name or a path (.npy/.npz array or image
folder). Training only writes checkpoints; estimate R_L(D) afterwards with
``evaluation/eval_lb.py`` using the same flags (it finds the same run directory).

    python train/train_lb.py --dataset gaussian --data_dim 1000 \
        --model mlp --units 100000,100000,100000 --lamb 100 \
        --checkpoint_dir checkpoints/gaussian \
        --batchsize 1024 --num_Ck_samples 2 --last_step 3000 \
        --y_init quick --y_quick_topn 10 --lr 5.0e-4 -V

Train + evaluate a whole sweep from one config:
    python scripts/run_sweep.py --config configs/gaussian_lb.yaml
"""
import os

import torch

from rdsandwich.cli.common import load_dataset
from rdsandwich.cli.lower_bound import (
    build_model,
    build_train_parser,
    ck_config,
    finalize_args,
    run_dir,
)
from rdsandwich.data.base import build_loader
from rdsandwich.lower_bound.trainer import LowerBoundTrainer
from rdsandwich.utils.io import JsonlLogger, get_time_str
from rdsandwich.utils.torch_utils import get_device, seed_everything


def main():
    args = build_train_parser().parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)

    dataset = load_dataset(args, device)
    finalize_args(args, dataset)
    model = build_model(args).to(device)
    cfg = ck_config(args)

    save_dir = run_dir(args)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"record-{get_time_str()}.jsonl")
    print(f"Logging to {log_path}")

    if hasattr(dataset, "__len__") and len(dataset) < cfg.batchsize:
        raise SystemExit(f"Dataset has {len(dataset)} examples, fewer than --batchsize (k) {cfg.batchsize}.")
    num_workers = 0 if args.preload else args.num_workers
    loader = build_loader(dataset, cfg.batchsize, num_workers=num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    trainer = LowerBoundTrainer(
        model, loader, optimizer=optimizer, cfg=cfg, device=device,
        logger=JsonlLogger(log_path), ckpt_dir=save_dir, verbose=args.verbose,
    )
    trainer.train()
    print(f"Checkpoints -> {save_dir}")


if __name__ == "__main__":
    main()
