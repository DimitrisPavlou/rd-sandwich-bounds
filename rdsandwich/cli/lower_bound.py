"""Argument parsing, model construction and run naming for the lower-bound CLIs
(``train/train_lb.py`` and ``evaluation/eval_lb.py``).

Both CLIs accept the same data / model / C_k flags, so one sweep config (or one
bash loop) can drive training and then evaluation of the same runs.
"""
from __future__ import annotations

import argparse
import os

from ..lower_bound import RDLBTrainConfig
from ..models.lower_bound import build_log_u_model
from ..utils import config_dict_to_str
from .common import add_data_args, add_run_args, example_shape, int_list, new_parser


def add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("log-u model")
    g.add_argument("--model", default="mlp", choices=["mlp", "cnn"])
    g.add_argument("--units", type=int_list, default=[], help="Hidden widths (cnn: conv channels then dense).")
    g.add_argument("--hidden_units_per_dim", type=int, default=0,
                   help="If >0 and --units is empty, build --num_hidden_layers hidden layers of "
                        "(hidden_units_per_dim * data_dim) units, so a data_dim sweep auto-sizes "
                        "the net; the paper (A.5.2) uses 20 for the Gaussian LB.")
    g.add_argument("--num_hidden_layers", type=int, default=2)
    g.add_argument("--activation", default="selu")
    g.add_argument("--kernel_dims", type=int_list, default=[], help="cnn only.")
    g.add_argument("--in_channels", type=int, default=3, help="cnn only.")
    g.add_argument("--lamb", type=float, required=True, help="Slope of the R-D lower bound.")


def add_ck_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("C_k estimator / inner optimizer")
    g.add_argument("--batchsize", "-k", type=int, default=1024, help="k")
    g.add_argument("--num_Ck_samples", "-M", type=int, default=1, help="M")
    g.add_argument("--y_init", default="quick", choices=["quick", "exhaustive"])
    g.add_argument("--y_quick_topn", type=int, default=10)
    g.add_argument("--y_steps", type=int, default=500)
    g.add_argument("--y_tol", type=float, default=1e-6)
    g.add_argument("--y_lr", type=float, default=1e-2)
    g.add_argument("--y_sequential", action="store_true",
                   help="Use the per-candidate sequential inner optimizer instead of the "
                        "(default) vectorized one.")
    g.add_argument("--cand_chunk", type=int, default=None,
                   help="Vectorized optimizer only: process the candidates in chunks of this "
                        "size to bound peak memory on high-dim/image data.")
    g.add_argument("--chunksize", type=int, default=None,
                   help="Chunk the pairwise-MSE computation to bound peak memory.")


def finalize_args(args, dataset) -> None:
    """Fill in values that depend on the data (in place)."""
    if args.data_dim is None:
        shape = example_shape(dataset)
        # mlp: flat vectors; cnn: square images, data_dim is the side length.
        args.data_dim = shape[0] if args.model == "mlp" else shape[-1]
    if not args.units and args.hidden_units_per_dim > 0:
        args.units = [args.hidden_units_per_dim * args.data_dim] * args.num_hidden_layers


def build_model(args):
    return build_log_u_model(
        args.model, args.data_dim, args.units,
        activation=args.activation, in_channels=args.in_channels, kernel_dims=args.kernel_dims,
    )


def ck_config(args, **overrides) -> RDLBTrainConfig:
    """``RDLBTrainConfig`` from the parsed flags (train-only fields when present)."""
    fields = dict(
        lamb=args.lamb, batchsize=args.batchsize, num_Ck_samples=args.num_Ck_samples,
        y_steps=args.y_steps, y_lr=args.y_lr, y_tol=args.y_tol, y_init=args.y_init,
        y_quick_topn=args.y_quick_topn, chunksize=args.chunksize,
        y_sequential=args.y_sequential, cand_chunk=args.cand_chunk,
    )
    for name in ("last_step", "lr", "beta", "log_E_Ck_max_delta", "checkpoint_interval"):
        if hasattr(args, name):
            fields[name] = getattr(args, name)
    fields.update(overrides)
    return RDLBTrainConfig(**fields)


def get_runname(args) -> str:
    return config_dict_to_str(
        vars(args), record_keys=("data_dim", "model", "units", "lamb", "batchsize", "seed"),
        prefix="rdlb",
    )


def run_dir(args) -> str:
    return os.path.join(args.checkpoint_dir, get_runname(args))


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def build_train_parser() -> argparse.ArgumentParser:
    p = new_parser("Train an R-D lower-bound log-u model (Algorithm 1).")
    add_run_args(p)
    add_data_args(p)
    add_model_args(p)
    add_ck_args(p)
    g = p.add_argument_group("optimization")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--last_step", type=int, default=3000)
    g.add_argument("--checkpoint_interval", type=int, default=1000)
    g.add_argument("--beta", type=float, default=0.2, help="EMA fraction retained for log_alpha.")
    g.add_argument("--log_E_Ck_max_delta", type=float, default=1.0)
    return p


def build_eval_parser() -> argparse.ArgumentParser:
    p = new_parser("Estimate R_L(D) for a trained lower-bound model (exhaustive global optimizer).")
    add_run_args(p)
    add_data_args(p)
    add_model_args(p)
    add_ck_args(p)
    p.set_defaults(y_init="exhaustive", num_Ck_samples=5)
    g = p.add_argument_group("evaluation")
    g.add_argument("--ckpt", default=None,
                   help="Checkpoint to evaluate (default: newest in the run directory).")
    return p
