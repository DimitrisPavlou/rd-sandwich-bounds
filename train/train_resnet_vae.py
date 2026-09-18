#!/usr/bin/env python
"""CLI for the image R-D upper bound (ResNetVAE), torch port of
``resnet_vae.py``'s train subcommand.

Natural images (matching README.md's Sec. 6 command, at reduced scale):

    python train/train_resnet_vae.py --dataset data/my_coco_train2017 \
        --img_dim 256 --latent_channels 4,8,16,32,64,128 --num_filters 256 \
        --lambda 0.01 --checkpoint_dir checkpoints/img_compression \
        --batchsize 8 --epochs 600 --steps_per_epoch 1000

GAN-generated images (Sec. 6.3; requires `pip install pytorch-pretrained-biggan`):

    python train/train_resnet_vae.py --dataset basenji --data_dim 4 \
        --img_dim 128 --flat_z0 --latent_channels 4,8,16,32,64,128 \
        --lambda 1e-4 --checkpoint_dir checkpoints/gan --batchsize 8 --epochs 600
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdsandwich.resnet_vae import ResNetVAE, ResNetVAEConfig
from rdsandwich.utils import config_dict_to_str, get_device, get_time_str, save_checkpoint, seed_everything, JsonlLogger


def get_source(args, device):
    if args.dataset in ("basenji",):
        from rdsandwich.biggan import BigGANSource
        return BigGANSource(args.dataset, intrinsic_dim=args.data_dim, device=device)
    from rdsandwich.dataloader import ImageFolderSource
    return ImageFolderSource(args.dataset, patchsize=args.img_dim, device=device,
                             preload=getattr(args, "preload", False))


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--verbose", "-V", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--device", default=None)

    p.add_argument("--dataset", required=True, help="Image folder / glob, or 'basenji' for the BigGAN source.")
    p.add_argument("--data_dim", type=int, default=None, help="Intrinsic dim, only used for the BigGAN source.")
    p.add_argument("--img_dim", type=int, default=128)
    p.add_argument("--latent_channels", type=lambda s: [int(i) for i in s.split(",")], default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--num_filters", type=int, default=256)
    p.add_argument("--flat_z0", action="store_true")
    p.add_argument("--lambda", type=float, default=1e-4, dest="lmbda")

    p.add_argument("--batchsize", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps_per_epoch", type=int, default=1000)
    p.add_argument("--preload", action="store_true",
                   help="Decode all images into RAM (uint8) once so sample() crops from "
                        "memory instead of re-decoding from disk every step.")
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)

    cfg = ResNetVAEConfig(
        img_dim=args.img_dim, latent_channels=args.latent_channels, num_filters=args.num_filters,
        lmbda=args.lmbda, flat_z0=args.flat_z0,
    )
    model = ResNetVAE(cfg).to(device)
    source = get_source(args, device)

    runname = config_dict_to_str(vars(args), record_keys=("dataset", "img_dim", "lmbda"), prefix="resnet_vae")
    save_dir = os.path.join(args.checkpoint_dir, runname)
    os.makedirs(save_dir, exist_ok=True)
    logger = JsonlLogger(os.path.join(save_dir, f"record-{get_time_str()}.jsonl"))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        running = {"loss": 0.0, "rate": 0.0, "mse": 0.0}
        for _ in range(args.steps_per_epoch):
            x = source.sample(args.batchsize)
            optimizer.zero_grad()
            loss, rate, mse = model.get_losses(x)
            loss.backward()
            optimizer.step()
            running["loss"] += loss.item()
            running["rate"] += rate.item()
            running["mse"] += mse.item()
        for k in running:
            running[k] /= args.steps_per_epoch
        if args.verbose:
            print(f"epoch {epoch}: {running}")
        logger.log({"epoch": epoch, **running})
    logger.close()

    ckpt_path = os.path.join(save_dir, f"ckpt-lmbda={args.lmbda}-epoch={args.epochs}.pt")
    save_checkpoint(ckpt_path, model, optimizer)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
