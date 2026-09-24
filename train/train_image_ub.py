#!/usr/bin/env python
"""CLI for the IMAGE R-D upper bound (paper Sec. 6.4): trains and evaluates the
two proposed beta-VAEs -- the hierarchical ResNet-VAE (``rdsandwich.resnet_vae``)
and the Minnen & Singh 2020 beta-VAE (``rdsandwich.ms2020_vae``).

Design (Option B): the *training* half reuses the existing, model-agnostic
``UpperBoundTrainer`` (its ``train_step`` just calls ``model.get_losses(x)``,
which both image models implement) plus ``ImageFolderSource`` + ``build_loader``.
The only genuinely new piece is the *evaluation* half: a full-image loop over the
Kodak/Tecnick test sets (variable-size images, reflect-padded to a multiple of
the downsampling factor, per-image bpp/PSNR, uint8 reconstruction cast), which
writes an ``rdub-model=...-lambda=...-dataset=....npz`` for ``evaluation/plot_qr.py``.

Train (one lambda; see configs/natural_images_*_train.yaml for the sweep):

    python train/train_image_ub.py --model resnet_vae --command train \
        --dataset data/my_coco_train2017 --latent_channels 4,8,16,32,64,128 \
        --ar_prior_levels 4 --ar_slices 8 --num_filters 256 --patchsize 256 \
        --lambda 0.01 --checkpoint_dir checkpoints/img_compression \
        --batchsize 8 --epochs 600 --steps_per_epoch 10000 --warmup 400 --patience 20

Evaluate on Kodak/Tecnick (loads the matching training checkpoint by arch+lambda):

    python train/train_image_ub.py --model resnet_vae --command eval \
        --dataset data/kodak --latent_channels 4,8,16,32,64,128 --ar_prior_levels 4 \
        --ar_slices 8 --num_filters 256 --lambda 0.01 \
        --checkpoint_dir checkpoints/img_compression --results_dir results/img_compression
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch._dynamo  # noqa: F401 (for torch._dynamo.config, used to raise the recompile limit)
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdsandwich.dataloader import ImageFolderSource, ImagePatchDataset
from rdsandwich.resnet_vae import ResNetVAE, ResNetVAEConfig
from rdsandwich.ms2020_vae import MS2020VAE, MS2020VAEConfig
from rdsandwich.upper_bound import UpperBoundTrainer
from rdsandwich.utils import (
    JsonlLogger, WarmupReduceLROnPlateau, get_device, get_time_str,
    latest_checkpoint, load_checkpoint, seed_everything,
)


# --------------------------------------------------------------------------- #
# Model / run-name construction
# --------------------------------------------------------------------------- #
def build_model(args):
    if args.model == "resnet_vae":
        cfg = ResNetVAEConfig(
            img_dim=args.patchsize, latent_channels=args.latent_channels,
            num_filters=args.num_filters, ar_prior_levels=args.ar_prior_levels,
            ar_slices=args.ar_slices, flat_z0=args.flat_z0,
            maf_units=args.maf_units, maf_stacks=args.maf_stacks, lmbda=args.lmbda,
            scale_min=args.scale_min,
        )
        return ResNetVAE(cfg), 2 ** len(args.latent_channels)
    if args.model == "ms2020_vae":
        cfg = MS2020VAEConfig(
            latent_depth=args.latent_depth, hyperprior_depth=args.hyperprior_depth,
            num_filters=args.num_filters, num_slices=args.num_slices,
            max_support_slices=args.max_support_slices, lmbda=args.lmbda,
        )
        return MS2020VAE(cfg), 64  # analysis(16x) * hyper(4x)
    raise ValueError(f"Unknown --model {args.model!r}")


def get_runname(args):
    """Checkpoint dir name; depends on model+arch+lambda only (NOT dataset/command),
    so `eval` reconstructs the same name as the `train` run that produced it."""
    parts = [f"rdub-model={args.model}", f"lambda={args.lmbda:g}"]
    if args.model == "resnet_vae":
        parts += [f"F={args.num_filters}",
                  "C=" + "_".join(str(c) for c in args.latent_channels),
                  f"arlv={args.ar_prior_levels}", f"arsl={args.ar_slices}"]
    else:
        parts += [f"ld={args.latent_depth}", f"hd={args.hyperprior_depth}",
                  f"ns={args.num_slices}", f"F={args.num_filters}"]
    return "-".join(parts)


def dataset_shortname(path):
    p = path.rstrip("/").lower()
    if "kodak" in p:
        return "kodak"
    if "tecnick" in p:
        return "tecnick"
    return os.path.basename(path.rstrip("/")) or "eval"


# --------------------------------------------------------------------------- #
# Train (reuses UpperBoundTrainer)
# --------------------------------------------------------------------------- #
def run_train(args, device):
    model, _ = build_model(args)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {pytorch_total_params}")
    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # autotune convs for the fixed patch size

    # Finite, map-style dataset: one random crop per image, a full pass per epoch
    # (traditional `for x in dataloader`), so every image is seen each epoch.
    dataset = ImagePatchDataset(args.dataset, patchsize=args.patchsize,
                                max_images=args.max_images, preload=args.preload)
    # Preloaded images live in RAM as uint8 and __getitem__ is just a crop, so there
    # is no decode work to parallelise -- run in-process (num_workers=0) to keep a
    # single resident copy instead of one per worker.
    num_workers = 0 if args.preload else args.num_workers
    # With torch.compile, keep the batch size CONSTANT (drop the size-mismatched last
    # batch) so dynamo doesn't recompile the graph for the partial final batch. The
    # model already runs shared sub-modules at several spatial resolutions per forward,
    # which alone approaches dynamo's recompile limit -- see the bump below.
    drop_last = bool(args.compile)
    loader = DataLoader(
        dataset, batch_size=args.batchsize, shuffle=True, drop_last=drop_last,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    steps_per_epoch = (len(dataset) // args.batchsize) if drop_last else math.ceil(len(dataset) / args.batchsize)

    save_dir = os.path.join(args.checkpoint_dir, get_runname(args))
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"record-{get_time_str()}.jsonl")
    ckpt_path = os.path.join(save_dir, f"ckpt-lambda={args.lmbda:g}.pt")
    print(f"{len(dataset)} images -> {steps_per_epoch} steps/epoch (batch {args.batchsize})")
    print(f"Logging to {log_path}\nCheckpoints -> {save_dir}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = WarmupReduceLROnPlateau(
        optimizer, factor=0.5, patience=args.patience, warmup=args.warmup, min_lr=1e-6,
    )

    # torch.compile fuses the many small conv/GDN/AR/deep-factorized kernels for a
    # large speedup (recovering the graph-execution advantage the authors got from
    # TensorFlow/XLA). We compile the *forward* and drive it from train_step, but
    # keep the ORIGINAL module in the trainer so checkpoints/params stay clean
    # (torch.compile shares parameters, so the optimizer still updates the same
    # tensors the compiled graph reads).
    #
    # The ResNet-VAE/ms2020 forwards call shared sub-modules (conv blocks, channel-AR
    # transforms, deep-factorized prior) at MANY spatial resolutions in one pass, so
    # dynamo must specialize a graph per resolution. That's a finite, fixed set, but
    # it exceeds dynamo's default recompile_limit (8) -> it would give up and fall
    # back to eager with noisy warnings. Raise the limit so every resolution compiles
    # once at startup (a few minutes), then steady state is fast.
    if args.compile:
        torch._dynamo.config.recompile_limit = 128
    fwd = torch.compile(model) if args.compile else model
    cl = args.channels_last

    class _Trainer(UpperBoundTrainer):
        def train_step(self, x):
            if cl:
                x = x.to(memory_format=torch.channels_last)
            out = fwd(x)
            return out["loss"], {"rate": out["bpp"].item(), "mse": out["mse"].item()}

        def checkpoint_extra(self):
            return {"cfg": vars(model.cfg), "model": args.model}

        def diagnostic_forward(self, x):
            # Eager (uncompiled) model so activation hooks fire; per-level latent stats.
            if cl:
                x = x.to(memory_format=torch.channels_last)
            out = model(x, return_stats=True) if args.model == "resnet_vae" else model(x)
            return out.get("stats", {})

    trainer = _Trainer(
        model, loader, optimizer=optimizer, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch, device=device, scheduler=scheduler,
        logger=JsonlLogger(log_path), ckpt_path=ckpt_path,
        grad_clip=args.grad_clip, amp=args.amp, max_nonfinite_skips=args.skip_nonfinite,
        checkpoint_interval=args.checkpoint_interval, resume=args.resume,
        verbose=args.verbose,
    )
    trainer.train()
    print(f"Saved checkpoint to {ckpt_path}")


# --------------------------------------------------------------------------- #
# Eval (the new full-image loop)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_eval(args, device):
    model, factor = build_model(args)
    model = model.to(device)

    save_dir = os.path.join(args.checkpoint_dir, get_runname(args))
    ckpt = args.ckpt or latest_checkpoint(save_dir)
    if ckpt is None:
        raise SystemExit(f"No checkpoint found in {save_dir!r}; train this run first.")
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()
    print(f"Loaded {ckpt}")

    source = ImageFolderSource(args.dataset, patchsize=None, device=device)
    bpps, mses, psnrs = [], [], []
    for x in source.all_images():                 # [1, 3, H, W] in [0, 255]
        x = x.to(device)
        _, _, H, W = x.shape
        pad_h, pad_w = (-H) % factor, (-W) % factor
        xp = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x
        out = model(xp)
        x_hat = out["x_hat"][:, :, :H, :W]
        if not args.no_cast_xhat:                  # discretized decoder omega (uint8)
            x_hat = torch.round(torch.clamp(x_hat, 0.0, 255.0))
        else:
            x_hat = torch.clamp(x_hat, 0.0, 255.0)
        mse = torch.mean((x - x_hat) ** 2).item()
        bpp = out["bits"].sum().item() / (H * W)   # bits over ORIGINAL pixel count
        bpps.append(bpp)
        mses.append(mse)
        psnrs.append(20 * math.log10(255.0) - 10 * math.log10(max(mse, 1e-12)))

    bpps, mses, psnrs = map(np.array, (bpps, mses, psnrs))
    dsname = dataset_shortname(args.dataset)
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(
        args.results_dir, f"rdub-model={args.model}-lambda={args.lmbda:g}-dataset={dsname}.npz")
    np.savez(out_path, bpp=bpps, mse=mses, psnr=psnrs)
    print(f"[{dsname}] n={len(bpps)}  bpp={bpps.mean():.4f}  mse={mses.mean():.4f}  "
          f"psnr={psnrs.mean():.4f}  ->  {out_path}")


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", required=True, choices=["resnet_vae", "ms2020_vae"])
    p.add_argument("--command", required=True, choices=["train", "eval"])
    p.add_argument("--dataset", required=True,
                   help="Train: COCO image folder. Eval: Kodak/Tecnick image folder.")
    p.add_argument("--checkpoint_dir", default="checkpoints/img_compression")
    p.add_argument("--results_dir", default="results/img_compression")
    p.add_argument("--lambda", type=float, default=0.01, dest="lmbda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", "-V", action="store_true")

    # shared architecture
    p.add_argument("--num_filters", type=int, default=256)
    # resnet_vae
    p.add_argument("--latent_channels", type=lambda s: [int(i) for i in s.split(",") if i],
                   default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--ar_prior_levels", type=int, default=4)
    p.add_argument("--ar_slices", type=int, default=8)
    p.add_argument("--flat_z0", action="store_true")
    p.add_argument("--scale_min", type=float, default=1e-5,
                   help="Floor on every Gaussian latent scale (numerical stability).")
    p.add_argument("--maf_units", type=lambda s: [int(i) for i in s.split(",") if i], default=[32, 16])
    p.add_argument("--maf_stacks", type=int, default=3)
    # ms2020_vae
    p.add_argument("--latent_depth", type=int, default=320)
    p.add_argument("--hyperprior_depth", type=int, default=192)
    p.add_argument("--num_slices", type=int, default=10)
    p.add_argument("--max_support_slices", type=int, default=5)

    # optimization (train)
    p.add_argument("--patchsize", type=int, default=256)
    p.add_argument("--batchsize", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes.")
    p.add_argument("--preload", action="store_true",
                   help="Decode the whole dataset into RAM (uint8) once at startup so no "
                        "epoch re-decodes from disk; forces num_workers=0. Best for sets "
                        "that fit in memory (~sum(3*H*W) bytes).")
    p.add_argument("--max_images", type=int, default=None,
                   help="Cap the number of training images (e.g. for smoke tests); default: all.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=600,
                   help="Full passes over the dataset (every image once per epoch).")
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--grad_clip", type=float, default=None,
                   help="Clip the global grad 2-norm to this value (on unscaled grads under --amp).")
    p.add_argument("--skip_nonfinite", type=int, default=0,
                   help="Skip up to this many consecutive steps with non-finite gradients "
                        "(each one still writes an explosion report); 0 = abort on the first.")
    p.add_argument("--amp", action="store_true", help="Mixed-precision (fp16 autocast + grad scaler).")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model (large speedup; first step compiles, ~1-2 min).")
    p.add_argument("--channels_last", action="store_true",
                   help="channels_last memory format (helps tensor-core GPUs; small on Turing).")
    p.add_argument("--checkpoint_interval", type=int, default=10)
    p.add_argument("--resume", action="store_true")

    # eval
    p.add_argument("--ckpt", default=None, help="Explicit checkpoint path (else newest in run dir).")
    p.add_argument("--no_cast_xhat", action="store_true",
                   help="Don't round the reconstruction to uint8 (clip only) -- for continuous sources.")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    if args.verbose:
        print(device)
    if args.command == "train":
        run_train(args, device)
    else:
        run_eval(args, device)


if __name__ == "__main__":
    main()
