#!/usr/bin/env python
"""Train an R-D UPPER-BOUND model: the MLP beta-VAE (vector data) or an image
beta-VAE (ResNet-VAE, Minnen & Singh 2020 VAE).

Every model is trained by the same ``UpperBoundTrainer``, which only calls
``model.get_losses(x)``; ``--dataset`` is a synthetic source name or a path
(.npy/.npz array or image folder). See ``rdsandwich/cli/upper_bound.py`` for
how to add a model. Evaluate afterwards with ``evaluation/eval_ub.py``.

Gaussian, n=1000, one (latent_dim, lambda) combo (sweep: configs/gaussian/mlp_vae_train_ub.yaml):

    python train/train_ub.py --model mlp_vae \
        --dataset gaussian --gparams_path data/gaussian/gaussian_params-dim=1000.npz \
        --latent_dim 1000 --prior_type gmm_1 --decoder_units 0 --lambda 10 \
        --checkpoint_dir checkpoints/gaussian \
        --epochs 80 --steps_per_epoch 1000 --lr 5.0e-4 --batchsize 64

Natural images, one lambda (sweep: configs/images/resnet_vae_train_ub.yaml):

    python train/train_ub.py --model resnet_vae \
        --dataset data/my_coco_train2017 --patchsize 256 \
        --latent_channels 4,8,16,32,64,128 --ar_prior_levels 4 --ar_slices 8 --num_filters 256 \
        --lambda 0.01 --checkpoint_dir checkpoints/img_compression \
        --batchsize 8 --lr 1.0e-4 --epochs 600 --lr_schedule plateau --warmup 400 --patience 20
"""
import os

import torch

from rdsandwich.cli.common import is_image_dataset, load_dataset
from rdsandwich.cli.upper_bound import (
    build_model,
    build_train_parser,
    checkpoint_filename,
    finalize_args,
    get_runname,
)
from rdsandwich.data.base import build_loader
from rdsandwich.upper_bound.trainer import UpperBoundTrainer
from rdsandwich.utils.io import JsonlLogger, get_time_str
from rdsandwich.utils.lr_schedulers import (
    WarmupReduceLROnPlateau,
    make_const_cos_scheduler,
    make_lr_scheduler,
)
from rdsandwich.utils.torch_utils import get_device, seed_everything


def make_scheduler(args, optimizer):
    if args.lr_schedule == "piecewise":
        return make_lr_scheduler(optimizer, args.epochs)
    if args.lr_schedule == "const-cos":
        return make_const_cos_scheduler(optimizer, args.epochs)
    if args.lr_schedule == "plateau":
        return WarmupReduceLROnPlateau(
            optimizer, factor=0.5, patience=args.patience, warmup=args.warmup, min_lr=1e-6,
        )
    return None


def main():
    args = build_train_parser().parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    if args.verbose:
        print(device)

    dataset = load_dataset(args, device)
    finalize_args(args, dataset)
    image_data = is_image_dataset(dataset)

    model = build_model(args).to(device)
    print(f"Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    if device.type == "cuda" and image_data:
        torch.backends.cudnn.benchmark = True  # autotune convs for the fixed patch size

    # Finite datasets: one full pass per epoch, dropping the last partial batch (a
    # constant batch size also keeps torch.compile from recompiling). Preloaded images
    # are already in RAM, so there is no decode work for worker processes to share.
    num_workers = 0 if args.preload else args.num_workers
    if hasattr(dataset, "__len__") and len(dataset) < args.batchsize:
        raise SystemExit(f"Dataset has {len(dataset)} examples, fewer than --batchsize {args.batchsize}.")
    loader = build_loader(dataset, args.batchsize, num_workers=num_workers,
                          pin_memory=image_data and device.type == "cuda")
    if hasattr(dataset, "__len__"):
        print(f"{len(dataset)} examples -> {len(loader)} steps/epoch (batch {args.batchsize}) "
              f"-> {len(loader) * args.epochs} steps in {args.epochs} epochs")

    save_dir = os.path.join(args.checkpoint_dir, get_runname(args))
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"record-{get_time_str()}.jsonl")
    ckpt_path = os.path.join(save_dir, checkpoint_filename(args))
    print(f"Logging to {log_path}\nCheckpoints -> {save_dir}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = UpperBoundTrainer(
        model, loader, optimizer=optimizer, epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch, device=device,
        scheduler=make_scheduler(args, optimizer), logger=JsonlLogger(log_path),
        ckpt_path=ckpt_path, verbose=args.verbose, grad_clip=args.grad_clip, amp=args.amp,
        max_nonfinite_skips=args.skip_nonfinite, checkpoint_interval=args.checkpoint_interval,
        resume=args.resume, compile=args.compile, channels_last=args.channels_last,
        ema_decay=args.ema, ema_warmup=args.ema_warmup,
        checkpoint_metadata={"model": args.model},
    )
    trainer.train()
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
