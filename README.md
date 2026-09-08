# rdsandwich (PyTorch)

A PyTorch re-implementation of

> Yibo Yang, Stephan Mandt. **Towards Empirical Sandwich Bounds on the
> Rate-Distortion Function.** ICLR 2022. https://arxiv.org/abs/2111.12166

ported from the original TensorFlow / `tensorflow-compression` codebase
(`RD-sandwich-master`), and reorganized into an installable library +
CLI scripts, so the three core algorithms can be reused, tested, and scaled
independently of any one experiment.

## Install

```bash
cd rd_reproduce
pip install -e .                 # core (Gaussian/banana/physics/speech experiments)
pip install -e ".[images]"       # + compressai, pytorch-pretrained-biggan (GAN/image experiments)
pip install -e ".[dev]"          # + pytest, matplotlib
```

Requires Python >= 3.9, PyTorch >= 2.0. Run the test suite with:

```bash
pytest tests/ -q
```

## Project layout

```
rdsandwich/                  installable package
  models/                    generic neural-network classes only
    mlp.py                   make_mlp / get_activation / GDN
    conv.py                  get_convnet (lower-bound image log-u model)
    flows.py                 MADE-based Masked Autoregressive Flow (flow prior)
  dataloader/                data sources + the DataLoader factory
    base.py                  Source, build_loader, InfiniteBatchDataset
    gaussian.py              Gaussian source, analytical R(D), param generator
    banana.py                2D "banana" source (+ n-d embeddings)
    array.py                 physics/speech .npy/.npz datasets
    image.py                 image-folder / patch-sampling source
  utils/                     shared infrastructure + classical approaches
    io.py                    jsonl logging, checkpointing, run-naming
    torch_utils.py           seeding, device, numeric helpers
    trainer.py               BaseTrainer (the shared training loop)
    ba.py                    Blahut-Arimoto reference algorithm
  upper_bound/               R-D UPPER BOUND
    config.py                RDUBConfig
    model.py                 RDUBModel (beta-VAE) + priors
    trainer.py               UpperBoundTrainer + (D,R) evaluator + LR schedule
  lower_bound/               R-D LOWER BOUND
    config.py                RDLBTrainConfig
    model.py                 build_log_u_model (log-u network factory)
    algorithm.py             batch_mse / compute_Ck_obj / optimize_y (C_k estimator)
    trainer.py               LowerBoundTrainer (overrides train()) + est_R_ evaluator
  resnet_vae.py              hierarchical conv-VAE for image UB (pending)
  compression_baselines.py   CompressAI mbt2018 / ms2020 baselines (pending)
  biggan.py                  BigGAN-backed GAN-image source (pending)
  config.py                  YAML sweep loader (expand_sweep / params_to_argv)
train/                       training CLIs (train_rdub / train_rdlb / train_resnet_vae)
evaluation/                  plotting + evaluation (plot_rdub.py)
scripts/                     misc CLIs (gen_gaussian_params, prepare_imgs, run_ba, run_sweep)
configs/                     YAML experiment configs (parameter sweeps)
data/                        generated / prepared data
tests/                       pytest suite
```

> Refactor status: the **upper and lower bounds** have been reorganized into
> `upper_bound/` and `lower_bound/` (each: config / model / trainer) on top of
> the shared `models/`, `dataloader/`, and `utils/` subpackages. Both trainers
> subclass `utils/trainer.py`'s `BaseTrainer` — `UpperBoundTrainer` fills its
> `train_step` hook, while `LowerBoundTrainer` overrides `train()` for the
> step-based Algorithm-1 loop. The image models (`resnet_vae.py`, `biggan.py`,
> `compression_baselines.py`) still live as flat modules pending their own slice.

## Mapping from the original TensorFlow repo

| Original file | Ported to | Notes |
|---|---|---|
| `rdub_mlp.py` | `rdsandwich/upper_bound/`, `train/train_rdub.py` | Full port: `gaussian`/`gmm_k`/`gsm_k`/`lmm_k`/`lsm_k`/`maf`/`std_gaussian` priors, optional decoder (Z==Y support), MLP encoder/decoder. The `'deep'` (DeepFactorized) prior and `posterior_type='uniform'` (used only for the NTC quantization baseline) are **not ported** — see `compression_baselines.py` for an NTC-equivalent baseline instead. |
| `rdlb.py` | `rdsandwich/lower_bound/`, `train/train_rdlb.py` | Full port of `compute_Ckobj`/`batch_mse`/`optimize_y`/the Algorithm-1 outer loop/`est_R_`. `--anneal_lamb` (single-run lambda sweep) is not ported; run one `--lamb` per invocation, as `experiments/*.sh` does. |
| `ba.py` | `rdsandwich/utils/ba.py`, `scripts/run_ba.py` | Near-verbatim port — the original was already NumPy/SciPy only. |
| `nn_models.py` | `rdsandwich/models/` (`mlp.py`, `conv.py`) | `make_mlp`/`get_activation`/GDN in `mlp.py`; `get_convnet` in `conv.py`. |
| `ntc_sources.py` | `rdsandwich/dataloader/` | `get_banana`/`get_nd_banana` -> `BananaSource`/`NdBananaEmbedder` (in `banana.py`), reproducing the same sequence of (inverted) transforms; `build_loader` turns any source into a `DataLoader`. |
| `gen_gaussian_params.py` | `scripts/gen_gaussian_params.py` | Direct port. |
| `prepare_imgs.py` | `scripts/prepare_imgs.py` | Direct port (Pillow instead of `tf.image`). |
| `resnet_vae.py` | `rdsandwich/resnet_vae.py`, `train/train_resnet_vae.py` | **Simplified port** — a ladder-VAE with the same level structure (strided-conv downsampling, per-level factorized-Gaussian posterior, optional MAF on the flattened top latent for `--flat_z0`) but *not* the original's full bidirectional-inference pass or channel-wise-autoregressive hyperprior. Good for reproducing the qualitative R-D upper bound shape on GAN images at moderate scale; for a bit-exact match to the paper's Kodak/Tecnick numbers, extend this scaffold with the bidirectional inference pass level-by-level. |
| `mbt2018.py`, `ms2020.py`, `biggan.py` | `rdsandwich/compression_baselines.py`, `rdsandwich/biggan.py` | **Not re-derived from scratch.** These wrap [CompressAI](https://github.com/InterDigitalInc/CompressAI) (PyTorch reference implementations of Minnen et al. 2018 and Minnen & Singh 2020) and `pytorch-pretrained-biggan` (a PyTorch port of the same DeepMind BigGAN-deep-128 weights used in the original TF-Hub-based script). Re-deriving two full published architectures from the TF source would dominate the porting effort for no scientific benefit over using their standard, maintained PyTorch reference implementations. |
| `boilerplate.py` | — | Keras-training-loop plumbing (LR schedules, callbacks); superseded by `rdsandwich/utils/trainer.py`'s `BaseTrainer` and the per-bound trainer subclasses. |
| `utils.py` | `rdsandwich/utils/` (`io.py`, `torch_utils.py`) | jsonl logging (`get_json_logging_callback` -> `JsonlLogger`), `config_dict_to_str`, checkpoint helpers replacing `model.save_weights`/`load_weights`; re-exported from `rdsandwich.utils`. |

## Quickstart: reproduce the n=1000 Gaussian sandwich bound

```bash
# 1. Generate the (fixed) random Gaussian source used in the paper
python scripts/gen_gaussian_params.py --save_dir data/gaussian --dim 1000

# 2. Upper bound: train a beta-VAE with Z == Y (no decoder), sweeping lambda
for lamb in 0.3 1 3 10 30 100 300; do
  python train/train_rdub.py --dataset gaussian --gparams_path data/gaussian/gaussian_params-dim=1000.npz \
      --data_dim 1000 --latent_dim 1000 --prior_type gmm_1 --decoder_units 0 \
      --lambda $lamb --checkpoint_dir checkpoints/gaussian \
      --epochs 80 --steps_per_epoch 1000 --lr 5e-4 --batchsize 64 -V
done
# (or drive the whole sweep from a YAML config — see "Running sweeps" below —
#  with:  python scripts/run_sweep.py --config configs/gaussian_ub_zy.yaml)

# 3. Lower bound: train a log-u MLP and (--eval_after) run the exhaustive-optimizer eval
python train/train_rdlb.py --dataset gaussian --data_dim 1000 --model mlp --units 100000,100000,100000 \
    --lamb 100 --checkpoint_dir checkpoints/gaussian \
    --command train --batchsize 1024 --num_Ck_samples 2 --last_step 3000 \
    --y_init quick --y_quick_topn 10 --lr 5e-4 --eval_after -V
# (or drive the whole lambda sweep from a config:
#  python scripts/run_sweep.py --config configs/gaussian_lb.yaml)

# to re-evaluate an existing checkpoint on its own:
python train/train_rdlb.py --dataset gaussian --data_dim 1000 --model mlp --units 100000,100000,100000 \
    --lamb 100 --checkpoint_dir checkpoints/gaussian \
    --command eval --ckpt checkpoints/gaussian/<run>/step=3000-....pt --y_init exhaustive --num_Ck_samples 5

# 4. Plot the sandwich figure from the trained checkpoints
python evaluation/plot_rdub.py --checkpoint_dir checkpoints/gaussian \
    --gparams_path data/gaussian/gaussian_params-dim=1000.npz --out results/gaussian_sandwich.png
```

See `configs/` for YAML sweep definitions and `experiments/` for the original
shell scripts covering the Gaussian, banana, particle-physics/speech,
GAN-image, and natural-image sweeps from the paper's README.

## Running hyperparameter sweeps from a YAML config

Instead of a GNU-`parallel` one-liner, sweeps can be declared in a YAML file and
run with `scripts/run_sweep.py`. For example, the paper's `n=1000` Gaussian
upper-bound sweep — originally

```bash
n=1000; parallel python rdub_mlp.py ... --latent_dim {1} --lambda {2} \
    ::: 400 500 600 800 ::: 0.3 1 3 10 30 100 300
```

is expressed by `configs/gaussian_ub.yaml`:

```yaml
script: train_rdub
fixed:                        # flags identical across every run
  dataset: gaussian
  data_dim: 1000
  prior_type: gmm_1
  decoder_units: 1000
  # ... (see the file for the rest)
sweep:                        # the ::: lists; the Cartesian product is run
  latent_dim: [400, 500, 600, 800]
  lambda: [0.3, 1, 3, 10, 30, 100, 300]
```

and run with:

```bash
python scripts/run_sweep.py --config configs/gaussian_ub.yaml   # 4 x 7 = 28 runs
python scripts/run_sweep.py --config configs/gaussian_ub.yaml --dry-run   # preview commands
python scripts/run_sweep.py --config configs/gaussian_ub.yaml -j 4        # 4 at a time
```

Each combination is launched as its own process (the same isolation as
`parallel`), with `-j/--jobs` controlling concurrency. The loader
(`rdsandwich.config`) is a small pure module: `expand_sweep` builds one params
dict per run and `params_to_argv` renders it into the `--flag value` argv the
`train_*.py` CLIs already accept. `run_sweep.py` drives the training CLIs
`train_rdub`, `train_rdlb`, and `train_resnet_vae` (all flat-flag). For the
lower bound, set `command: train` and `eval_after: true` in the config so each
run trains and then writes its `rd-*.npz` (see `configs/gaussian_lb.yaml`).

> Note on YAML floats: write `5.0e-4`, not `5e-4` — PyYAML parses `5e-4` as a
> string (a well-known quirk). `5.0e-4` and `0.0005` both parse as floats.

## Scaling this up

The original scripts were written for one run == one process == one GPU,
driven by GNU-`parallel`-style Cartesian-product shell one-liners. This
port keeps that same "one hyperparameter combo per process" granularity
(so the shell scripts in `experiments/` still work), but a few things make
it easier to actually scale:

- **Each algorithm is a plain, importable PyTorch training loop**
  (`train_rdub`, `train_rdlb`, `ResNetVAE.get_losses` + your own loop, or
  `compression_baselines.train_baseline`) with no Keras/`tf.function`
  tracing to fight with — drop them into a job-queue (Slurm array, Ray, or
  `torch.multiprocessing`) exactly as you would any other PyTorch script.
- **`RDLBTrainConfig.chunksize`** bounds peak memory in the pairwise-MSE
  inner optimization (`optimize_y`/`batch_mse`), the main memory bottleneck
  for the lower bound on large `k` or high-resolution images.
- **Multi-GPU / mixed precision**: the models are ordinary `nn.Module`s, so
  wrapping them in `torch.nn.parallel.DistributedDataParallel` or running
  under `torch.autocast` works out of the box for `rdub.py`/`resnet_vae.py`;
  `rdlb.py`'s inner `optimize_y` loop is inherently sequential per-batch
  but embarrassingly parallel *across* the `M` (`num_Ck_samples`) draws —
  see the `for _ in range(M)` loop in `train_rdlb`, which is the natural
  place to fan out across devices/processes.
- **Neural-compression baselines and BigGAN sampling now come from
  actively-maintained, GPU-optimized PyTorch packages** (CompressAI,
  `pytorch-pretrained-biggan`) instead of hand-rolled TF ports, so they
  benefit from upstream performance work for free.
- **jsonl logs** (`rdsandwich.utils.JsonlLogger`) are the same flat,
  appendable format as the original repo, so existing `utils.aggregate_*`
  style post-processing / plotting code needs only minor tweaks.

## Citing

If you use this code, please cite the original paper:

```bibtex
@inproceedings{yang2022towards,
  title={Towards Empirical Sandwich Bounds on the Rate-Distortion Function},
  author={Yang, Yibo and Mandt, Stephan},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

This repository is an independent PyTorch port and is not affiliated with
the original authors.
