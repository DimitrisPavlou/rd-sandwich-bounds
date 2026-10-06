"""Argument parsing, model construction and run naming for the upper-bound CLIs
(``train/train_ub.py`` and ``evaluation/eval_ub.py``).

To add a new upper-bound model:
  1. write it in ``rdsandwich/models/upper_bound/`` with
     ``get_losses(x) -> (loss, rate, distortion)`` (and, for full-image
     evaluation, a ``forward(x)`` returning ``x_hat`` and per-image ``bits``);
  2. add its name to ``MODELS``, its flags to ``add_model_args``, and a branch to
     ``build_model`` (plus ``downsampling_factor`` if it is an image model).
"""
from __future__ import annotations

import argparse
import os

from rdsandwich.models.upper_bound.ms2020_vae import MS2020VAE, MS2020VAEConfig
from rdsandwich.models.upper_bound.mlp_vae import RDUBConfig, RDUBModel, check_no_decoder
from rdsandwich.models.upper_bound.resnet_vae import ResNetVAE, ResNetVAEConfig
from rdsandwich.utils.io import config_dict_to_str
from rdsandwich.cli.common import add_data_args, add_run_args, example_shape, int_list, new_parser

MODELS = ("mlp_vae", "resnet_vae", "ms2020_vae")


def add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--model", required=True, choices=MODELS)
    g.add_argument("--lambda", type=float, default=0.01, dest="lmbda",
                   help="R-D trade-off: loss = rate + lambda * distortion.")

    g = p.add_argument_group("mlp_vae (vector data)")
    g.add_argument("--latent_dim", type=int, default=None, help="Forced to data_dim when Z == Y.")
    g.add_argument("--encoder_units", type=int_list, default=[])
    g.add_argument("--decoder_units", type=int_list, default=[],
                   help="'0' means no decoder network (Z == Y).")
    g.add_argument("--encoder_activation", default="softplus")
    g.add_argument("--decoder_activation", default="softplus")
    g.add_argument("--prior_type", default="std_gaussian",
                   help="std_gaussian | maf | gmm_<k> | gsm_<k> | lmm_<k> | lsm_<k>")
    g.add_argument("--posterior_type", default="gaussian")
    g.add_argument("--ar_hidden_units", type=int_list, default=[10, 10],
                   help="MADE hidden widths inside each MAF stack (prior_type=maf).")
    g.add_argument("--maf_stacks", type=int, default=0, help="MAF flow stacks (prior_type=maf).")
    g.add_argument("--rpd", action="store_true", help="Report rate per data dimension.")
    g.add_argument("--nats", action="store_true", help="Report rate in nats (else bits).")

    g = p.add_argument_group("resnet_vae / ms2020_vae (images)")
    g.add_argument("--num_filters", type=int, default=256)

    g = p.add_argument_group("resnet_vae")
    g.add_argument("--latent_channels", type=int_list, default=[4, 8, 16, 32, 64, 128])
    g.add_argument("--ar_prior_levels", type=int, default=4)
    g.add_argument("--ar_slices", type=int, default=8)
    g.add_argument("--scale_min", type=float, default=1e-5,
                   help="Floor on every Gaussian latent scale (numerical stability).")

    g = p.add_argument_group("ms2020_vae")
    g.add_argument("--latent_depth", type=int, default=320)
    g.add_argument("--hyperprior_depth", type=int, default=192)
    g.add_argument("--num_slices", type=int, default=10)
    g.add_argument("--max_support_slices", type=int, default=5)


def finalize_args(args, dataset) -> None:
    """Fill in values that depend on the model or the data (in place)."""
    if args.model == "mlp_vae":
        if args.data_dim is None:
            shape = example_shape(dataset)
            if len(shape) != 1:
                raise SystemExit(f"mlp_vae needs vector data, got examples of shape {shape}.")
            args.data_dim = shape[0]
        if check_no_decoder(args.decoder_units):
            if args.latent_dim not in (None, args.data_dim):
                print(f"Using Z=Y; resetting latent_dim={args.latent_dim} to data_dim={args.data_dim}")
            args.latent_dim = args.data_dim
        if args.latent_dim is None:
            raise SystemExit("mlp_vae needs --latent_dim (unless --decoder_units 0).")


def build_model(args):
    if args.model == "mlp_vae":
        return RDUBModel(RDUBConfig(
            data_dim=args.data_dim, latent_dim=args.latent_dim, lmbda=args.lmbda,
            encoder_units=args.encoder_units, decoder_units=args.decoder_units,
            encoder_activation=args.encoder_activation, decoder_activation=args.decoder_activation,
            prior_type=args.prior_type, posterior_type=args.posterior_type,
            ar_hidden_units=args.ar_hidden_units, maf_stacks=args.maf_stacks,
            rpd=args.rpd, nats=args.nats,
        ))
    if args.model == "resnet_vae":
        return ResNetVAE(ResNetVAEConfig(
            latent_channels=args.latent_channels, num_filters=args.num_filters,
            ar_prior_levels=args.ar_prior_levels, ar_slices=args.ar_slices,
            lmbda=args.lmbda, scale_min=args.scale_min,
        ))
    if args.model == "ms2020_vae":
        return MS2020VAE(MS2020VAEConfig(
            latent_depth=args.latent_depth, hyperprior_depth=args.hyperprior_depth,
            num_filters=args.num_filters, num_slices=args.num_slices,
            max_support_slices=args.max_support_slices, lmbda=args.lmbda,
        ))
    raise ValueError(f"Unknown --model {args.model!r}")


def downsampling_factor(args) -> int:
    """Total spatial downsampling of an image model (full images are padded to a multiple)."""
    if args.model == "resnet_vae":
        return 2 ** len(args.latent_channels)
    if args.model == "ms2020_vae":
        return 64  # analysis (16x) * hyper-analysis (4x)
    raise ValueError(f"{args.model!r} is not an image model")


def get_runname(args) -> str:
    """Run directory name. Depends on the model, its architecture and lambda only
    (not on the dataset or optimizer), so evaluation reconstructs the same name as
    the training run that produced the checkpoint."""
    if args.model == "mlp_vae":
        return config_dict_to_str(
            vars(args),
            record_keys=("data_dim", "latent_dim", "lmbda", "encoder_units", "decoder_units",
                         "prior_type", "posterior_type", "maf_stacks"),
            prefix="rdub",
        )
    parts = [f"rdub-model={args.model}", f"lambda={args.lmbda:g}"]
    if args.model == "resnet_vae":
        parts += [f"F={args.num_filters}",
                  "C=" + "_".join(str(c) for c in args.latent_channels),
                  f"arlv={args.ar_prior_levels}", f"arsl={args.ar_slices}"]
    else:
        parts += [f"ld={args.latent_depth}", f"hd={args.hyperprior_depth}",
                  f"ns={args.num_slices}", f"F={args.num_filters}"]
    return "-".join(parts)


def run_dir(args) -> str:
    return os.path.join(args.checkpoint_dir, get_runname(args))


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def build_train_parser() -> argparse.ArgumentParser:
    p = new_parser("Train an R-D upper-bound model (any model with get_losses(x)).")
    add_run_args(p)
    add_data_args(p)
    add_model_args(p)

    g = p.add_argument_group("optimization")
    g.add_argument("--batchsize", type=int, default=64)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--epochs", type=int, default=100,
                   help="Finite datasets: full passes over the data. Synthetic sources: "
                        "blocks of --steps_per_epoch steps.")
    g.add_argument("--steps_per_epoch", type=int, default=1000, help="Synthetic sources only.")
    g.add_argument("--lr_schedule", choices=["piecewise", "plateau", "constant"], default="piecewise",
                   help="piecewise: x0.2 at 50/75/87.5%% of the epochs. plateau: halve the LR "
                        "after --patience epochs without improvement, after --warmup epochs.")
    g.add_argument("--warmup", type=int, default=400, help="lr_schedule=plateau only.")
    g.add_argument("--patience", type=int, default=20, help="lr_schedule=plateau only.")
    g.add_argument("--grad_clip", type=float, default=None,
                   help="Clip the global grad 2-norm to this value (on unscaled grads under --amp).")
    g.add_argument("--skip_nonfinite", type=int, default=0,
                   help="Skip up to this many consecutive steps with non-finite gradients "
                        "(each one still writes an explosion report); 0 = abort on the first.")
    g.add_argument("--amp", action="store_true", help="Mixed precision (fp16 autocast + grad scaler).")
    g.add_argument("--compile", action="store_true",
                   help="torch.compile model.get_losses (large speedup for image models; "
                        "the first steps compile).")
    g.add_argument("--channels_last", action="store_true", help="channels_last memory format.")
    g.add_argument("--checkpoint_interval", type=int, default=None,
                   help="Also save a checkpoint every N epochs (default: only at the end).")
    g.add_argument("--resume", action="store_true",
                   help="Resume from the newest checkpoint in the run directory.")
    return p


def build_eval_parser() -> argparse.ArgumentParser:
    p = new_parser("Evaluate a trained R-D upper-bound model.")
    add_run_args(p)
    add_data_args(p)
    add_model_args(p)

    g = p.add_argument_group("evaluation")
    g.add_argument("--ckpt", default=None,
                   help="Checkpoint to evaluate (default: newest in the run directory).")
    g.add_argument("--results_dir", default="results")
    g.add_argument("--mode", choices=["auto", "full_image", "sampled"], default="auto",
                   help="full_image: per-image bpp/MSE/PSNR on whole images (Kodak/Tecnick). "
                        "sampled: (D, R) mean +/- 95%% CI over --num_batches batches. "
                        "auto: full_image for image folders, sampled otherwise.")
    g.add_argument("--batchsize", type=int, default=1024, help="sampled mode only.")
    g.add_argument("--num_batches", type=int, default=100, help="sampled mode only.")
    g.add_argument("--no_cast_xhat", action="store_true",
                   help="full_image mode: don't round the reconstruction to uint8 (clip only).")
    return p
