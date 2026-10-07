#!/usr/bin/env python
"""Evaluate a trained R-D UPPER-BOUND model (written by ``train/train_ub.py``).

Pass the same model flags (and lambda / checkpoint_dir) as for training: they
determine the run directory, whose newest checkpoint is loaded (or ``--ckpt``).

* ``--mode full_image`` (default for image folders): per-image bpp / MSE / PSNR
  on whole Kodak/Tecnick images (unrounded and 8-bit-rounded reconstructions),
  saved to ``<results_dir>/rdub-model=<key>-lambda=<lambda>-dataset=<name>.npz``,
  where ``<key>`` is the model plus any curve-defining tags (e.g.
  ``resnet_vae-range=pm1``; see ``model_key``). Read by ``evaluation/plot_qr.py``.
  For the variable-rate ``variable_rate_lossy_vae``, run once per ``--lambda`` on the same
  checkpoint.
* ``--mode sampled`` (default otherwise): (D, R) mean / std / 95% CI over
  ``--num_batches`` batches, saved to ``<results_dir>/<run name>-dataset=<name>.npz``.

    python evaluation/eval_ub.py --model resnet_vae --dataset data/kodak \
        --latent_channels 4,8,16,32,64,128 --ar_prior_levels 4 --ar_slices 8 --num_filters 256 \
        --lambda 0.01 --checkpoint_dir checkpoints/img_compression --results_dir results/img_compression
"""
import os

import numpy as np

from rdsandwich.cli.common import is_image_dataset, load_dataset
from rdsandwich.cli.upper_bound import (
    build_eval_parser,
    build_model,
    downsampling_factor,
    finalize_args,
    get_runname,
    model_key,
    run_dir,
)
from rdsandwich.data.datasets import dataset_name
from rdsandwich.upper_bound.evaluate import evaluate_full_images, evaluate_sampled
from rdsandwich.utils.torch_utils import get_device, seed_everything
from rdsandwich.utils.io import latest_checkpoint, load_checkpoint


def main():
    args = build_eval_parser().parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)

    dataset = load_dataset(args, device)
    finalize_args(args, dataset)
    model = build_model(args).to(device)

    ckpt = args.ckpt or latest_checkpoint(run_dir(args))
    if ckpt is None:
        raise SystemExit(f"No checkpoint found in {run_dir(args)!r}; train this run first.")
    extra = load_checkpoint(ckpt, model, map_location=device, use_ema=not args.no_ema)
    model.eval()
    weights = "EMA" if ("ema_state_dict" in extra and not args.no_ema) else "raw"
    print(f"Loaded {ckpt} ({weights} weights)")

    mode = args.mode
    if mode == "auto":
        mode = "full_image" if is_image_dataset(dataset) else "sampled"
    dsname = dataset_name(args.dataset)
    os.makedirs(args.results_dir, exist_ok=True)

    if mode == "full_image":
        if not is_image_dataset(dataset):
            raise SystemExit("--mode full_image needs an image-folder --dataset.")
        res = evaluate_full_images(model, dataset.all_images(), pad_factor=downsampling_factor(args),
                                   device=device)
        out_path = os.path.join(
            args.results_dir, f"rdub-model={model_key(args)}-lambda={args.lmbda:g}-dataset={dsname}.npz")
        np.savez(out_path, **res)
        print(f"[{dsname}] n={len(res['bpp'])}  bpp={res['bpp'].mean():.4f}  "
              f"mse={res['mse'].mean():.4f}  psnr={res['psnr'].mean():.4f}  "
              f"(8-bit: psnr={res['psnr_uint8'].mean():.4f})  ->  {out_path}")
    else:
        res = evaluate_sampled(model, dataset, args.batchsize, args.num_batches, device=device)
        out_path = os.path.join(args.results_dir, f"{get_runname(args)}-dataset={dsname}.npz")
        np.savez(out_path, **res)
        print(f"[{dsname}] rate={res['rate_mean']:.5g} (95% CI {res['rate_ci'][0]:.5g}..{res['rate_ci'][1]:.5g})  "
              f"mse={res['mse_mean']:.5g} (95% CI {res['mse_ci'][0]:.5g}..{res['mse_ci'][1]:.5g})  ->  {out_path}")


if __name__ == "__main__":
    main()
