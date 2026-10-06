# rdsandwich (PyTorch)

A PyTorch re-implementation of

> Yibo Yang, Stephan Mandt. **Towards Empirical Sandwich Bounds on the
> Rate-Distortion Function.** ICLR 2022. https://arxiv.org/abs/2111.12166

ported from the original TensorFlow / `tensorflow-compression` codebase
(`RD-sandwich-master`), and reorganized into an installable library + CLI
scripts, so the core algorithms can be reused, tested, and applied to new
models and data without touching the training or evaluation code.

## Install

```bash
pip install -e .                 # the rdsandwich package + its dependencies
pip install -e ".[dev]"          # + pytest, matplotlib
```

Requires Python >= 3.9, PyTorch >= 2.0. Run the test suite with:

```bash
pytest tests/ -q
```

The scripts import the `rdsandwich` package, so install it (editable) first.
The shell scripts in `experiments/` and `scripts/slurm/` instead put the repo on
`PYTHONPATH` themselves, so they also run in an environment without the install.

## Project layout

```
rdsandwich/                  installable package
  models/                    everything that gets TRAINED, grouped by bound
    upper_bound/             beta-VAEs; each has get_losses(x) -> (loss, rate, distortion)
      mlp_vae.py             RDUBConfig / RDUBModel: MLP beta-VAE for vector data (+ priors)
      resnet_vae.py          hierarchical ResNet-VAE for images (bidirectional inference)
      ms2020_vae.py          Minnen & Singh 2020 beta-VAE for images (channel-AR prior)
      _common.py             Gaussian-latent helpers shared by the image models
    lower_bound/
      log_u.py               build_log_u_model: mlp / cnn log-u networks
  layers/                    generic building blocks the models are made of
    mlp.py gdn.py conv.py flows.py deep_factorized.py channelwise_ar.py
  upper_bound/               HOW the upper bound is trained and evaluated
    trainer.py               UpperBoundTrainer (any get_losses model; compile / channels_last) + LR schedule
    evaluate.py              evaluate_sampled ((D, R) +/- 95% CI), evaluate_full_images (Kodak/Tecnick)
  lower_bound/               HOW the lower bound is trained and evaluated
    config.py                RDLBTrainConfig
    algorithm.py             batch_mse / compute_Ck_obj / optimize_y / run_optimize_y (C_k estimator)
    trainer.py               LowerBoundTrainer (Algorithm 1)
    evaluate.py              estimate_R_lower_bound (est_R_)
  data/                      get_dataset(spec) + build_loader, and the sources:
    gaussian.py banana.py    synthetic sources
    array.py                 .npy/.npz datasets (physics, speech, ...)
    image.py                 ImageFolderDataset (patches for training, whole images for eval)
  utils/                     BaseTrainer, logging/checkpointing, seeding, Blahut-Arimoto (ba.py)
  cli/                       argument parsing / model building / run naming shared by the CLIs
  sweep.py                   YAML sweep configs (expand_sweep / script_params / params_to_argv)
train/                       train_ub.py, train_lb.py
evaluation/                  eval_ub.py, eval_lb.py + plotting (plot_rdub, plot_rdlb, plot_qr, ...)
scripts/                     run_sweep, gen_gaussian_params, prepare_imgs, run_ba, slurm/
configs/                     YAML experiment configs; configs/templates/ documents every flag
experiments/                 shell scripts reproducing the paper's sweeps
docs/figures/                model diagrams
data/                        generated / prepared data (git-ignored)
tests/                       pytest suite
```

## How the pieces fit

| | Upper bound | Lower bound |
|---|---|---|
| models | `rdsandwich.models.upper_bound` | `rdsandwich.models.lower_bound` |
| trainer | `UpperBoundTrainer` | `LowerBoundTrainer` |
| evaluator | `evaluate_sampled` / `evaluate_full_images` | `estimate_R_lower_bound` |
| training CLI | `train/train_ub.py --model {mlp_vae,resnet_vae,ms2020_vae}` | `train/train_lb.py --model {mlp,cnn}` |
| evaluation CLI | `evaluation/eval_ub.py` | `evaluation/eval_lb.py` |

Training and evaluation are separate scripts. An evaluation CLI takes the same
model flags (and `--checkpoint_dir`, `--lambda`/`--lamb`) as training: they
determine the run directory, whose newest checkpoint is loaded (or pass `--ckpt`).

### Data: `--dataset`

Every CLI takes `--dataset`, which is one of

- `gaussian` or `banana`: a synthetic source (sampled on demand; needs `--data_dim`,
  or `--gparams_path` for a fixed non-standard Gaussian);
- a **path**, used exactly as given, relative or absolute, inside `data/` or
  anywhere else on disk (e.g. a large dataset on another drive):
  - a `.npy` / `.npz` file: rows are samples (`--data_dim` is inferred);
  - an image directory or glob: random `--patchsize` crops for training, whole
    images for full-image evaluation (`--preload` decodes them into RAM once).

### Adding a new upper-bound model

1. Write it in `rdsandwich/models/upper_bound/` as an `nn.Module` with
   `get_losses(x) -> (loss, rate, distortion)`. That is all
   `UpperBoundTrainer` needs. For full-image evaluation, `forward(x)` must also
   return a dict with `x_hat` and per-image `bits`. Optionally, a
   `diagnostics(x) -> dict` method adds model statistics to the trainer's
   explosion report (written when the loss or gradients become non-finite).
2. In `rdsandwich/cli/upper_bound.py`, add its name to `MODELS`, its flags to
   `add_model_args`, and a branch to `build_model` (plus `downsampling_factor`
   for an image model, and a run-name entry in `get_runname`).

The data loading, trainer, evaluators, sweep runner and plots then work unchanged.

## Quickstart: the n=1000 Gaussian sandwich bound

```bash
# 1. Generate the (fixed) random Gaussian source used in the paper
python scripts/gen_gaussian_params.py --save_dir data/gaussian --dim 1000

# 2. Upper bound: a beta-VAE with Z == Y (no decoder), one run per lambda
for lamb in 0.3 1 3 10 30 100 300; do
  python train/train_ub.py --model mlp_vae \
      --dataset gaussian --gparams_path data/gaussian/gaussian_params-dim=1000.npz \
      --prior_type gmm_1 --decoder_units 0 --lambda $lamb --checkpoint_dir checkpoints/gaussian \
      --epochs 80 --steps_per_epoch 1000 --lr 5e-4 --batchsize 64 -V
done
# (or the whole sweep:  python scripts/run_sweep.py --config configs/gaussian_ub_zy.yaml)

# 3. Lower bound: train a log-u MLP, then estimate R_L(D) with the exhaustive optimizer
lb="--dataset gaussian --data_dim 1000 --model mlp --units 100000,100000,100000 --lamb 100
    --batchsize 1024 --checkpoint_dir checkpoints/gaussian"
python train/train_lb.py $lb --num_Ck_samples 2 --last_step 3000 --y_init quick --y_quick_topn 10 --lr 5e-4 -V
python evaluation/eval_lb.py $lb --num_Ck_samples 5
# (or the whole lambda sweep, train + eval:  python scripts/run_sweep.py --config configs/gaussian_lb.yaml)

# 4. Plot the sandwich figure
python evaluation/plot_rdub.py --checkpoint_dir checkpoints/gaussian \
    --gparams_path data/gaussian/gaussian_params-dim=1000.npz --out results/gaussian_sandwich.png
```

Natural images (Sec. 6.4) work the same way, with an image folder as `--dataset`:

```bash
arch="--model resnet_vae --latent_channels 4,8,16,32,64,128 --ar_prior_levels 4 --ar_slices 8 --num_filters 256"
python train/train_ub.py $arch --lambda 0.01 --dataset /path/to/my_coco_train2017 --patchsize 256 \
    --batchsize 8 --lr 1e-4 --epochs 600 --lr_schedule plateau --warmup 400 --patience 20 \
    --checkpoint_dir checkpoints/img_compression --compile -V
python evaluation/eval_ub.py $arch --lambda 0.01 --dataset data/kodak \
    --checkpoint_dir checkpoints/img_compression --results_dir results/img_compression
python evaluation/plot_qr.py --dataset kodak --results_dir results/img_compression --out results/qr_kodak.png
```

See `configs/` for YAML sweep definitions and `experiments/` for shell scripts
covering the Gaussian, banana, particle-physics/speech, and natural-image sweeps.

## Running sweeps: YAML configs or bash

Sweeps can be declared in a YAML file and run with `scripts/run_sweep.py`. For
example, the paper's `n=1000` Gaussian upper-bound sweep, originally

```bash
n=1000; parallel python rdub_mlp.py ... --latent_dim {1} --lambda {2} \
    ::: 400 500 600 800 ::: 0.3 1 3 10 30 100 300
```

is `configs/gaussian_ub.yaml`:

```yaml
script: train_ub
fixed:                        # flags identical across every run
  model: mlp_vae
  dataset: gaussian
  gparams_path: data/gaussian/gaussian_params-dim=1000.npz
  prior_type: gmm_1
  decoder_units: 1000
  # ... (see the file for the rest)
sweep:                        # the ::: lists; the Cartesian product is run
  latent_dim: [400, 500, 600, 800]
  lambda: [0.3, 1, 3, 10, 30, 100, 300]
```

```bash
python scripts/run_sweep.py --config configs/gaussian_ub.yaml             # 4 x 7 = 28 runs
python scripts/run_sweep.py --config configs/gaussian_ub.yaml --dry-run   # print the commands
python scripts/run_sweep.py --config configs/gaussian_ub.yaml -j 4        # 4 at a time
```

Each combination runs as its own process, with `-j/--jobs` controlling
concurrency. `script` can also be a list, run in order for every combination,
with `script_args` for flags that only one of the scripts takes. This is how the
lower-bound configs train and then evaluate each run:

```yaml
script: [train_lb, eval_lb]
fixed: {...}                  # passed to both (they locate the same run directory)
script_args:
  train_lb: {num_Ck_samples: 2, y_init: quick, last_step: 3000, lr: 5.0e-4}
  eval_lb:  {num_Ck_samples: 5, y_init: exhaustive}
```

Configs are only a front-end to the CLIs: every YAML key is a `--flag`, so a
bash loop (`experiments/*.sh`), a Slurm array (`scripts/slurm/`) or the output
of `--dry-run` can drive any model the same way. `configs/templates/` lists
every flag of every script, and `tests/test_configs.py` checks that all configs
still parse.

> Note on YAML floats: write `5.0e-4`, not `5e-4` — PyYAML parses `5e-4` as a
> string (a well-known quirk). `5.0e-4` and `0.0005` both parse as floats.

## Scaling this up

The original scripts were written for one run == one process == one GPU. This
port keeps that granularity (one hyperparameter combination per process), and:

- **Each algorithm is a plain, importable PyTorch training loop** with no
  Keras/`tf.function` tracing, so runs drop into a job queue (Slurm array, Ray,
  `torch.multiprocessing`) like any other PyTorch script.
- **`--compile`, `--amp`, `--channels_last`** are trainer options, so every
  upper-bound model gets them; `--compile` gives the image models a large speedup.
- **`RDLBTrainConfig.chunksize` / `cand_chunk`** bound peak memory in the
  lower bound's pairwise-MSE inner optimization, its main memory bottleneck for
  large `k` or high-resolution images.
- **The lower bound's inner `optimize_y`** is sequential per batch but
  embarrassingly parallel across the `M` (`num_Ck_samples`) draws: the
  `for _ in range(M)` loop in `LowerBoundTrainer.train` is the natural place to
  fan out across devices.
- **jsonl logs** (`rdsandwich.utils.io.JsonlLogger`) use the same flat, appendable
  format as the original repo.

## Mapping from the original TensorFlow repo

| Original file | Ported to | Notes |
|---|---|---|
| `rdub_mlp.py` | `rdsandwich/models/upper_bound/mlp_vae.py`, `train/train_ub.py --model mlp_vae` | Full port: `gaussian`/`gmm_k`/`gsm_k`/`lmm_k`/`lsm_k`/`maf`/`std_gaussian` priors, optional decoder (Z==Y support), MLP encoder/decoder. The `'deep'` (DeepFactorized) prior and `posterior_type='uniform'` (used only for the NTC quantization baseline) are **not ported**. |
| `rdlb.py` | `rdsandwich/lower_bound/`, `train/train_lb.py`, `evaluation/eval_lb.py` | Full port of `compute_Ckobj`/`batch_mse`/`optimize_y`/the Algorithm-1 outer loop/`est_R_`. `--anneal_lamb` (single-run lambda sweep) is not ported; run one `--lamb` per invocation. |
| `ba.py` | `rdsandwich/utils/ba.py`, `scripts/run_ba.py` | Near-verbatim port — the original was already NumPy/SciPy only. |
| `nn_models.py` | `rdsandwich/layers/` (`mlp.py`, `conv.py`), `rdsandwich/models/lower_bound/log_u.py` | `make_mlp`/`get_activation`/GDN in `mlp.py`; `get_convnet` in `conv.py`. |
| `ntc_sources.py` | `rdsandwich/data/` | `get_banana`/`get_nd_banana` -> `BananaSource`/`NdBananaEmbedder` (in `banana.py`), reproducing the same sequence of (inverted) transforms. |
| `gen_gaussian_params.py` | `scripts/gen_gaussian_params.py` | Direct port. |
| `prepare_imgs.py` | `scripts/prepare_imgs.py` | Direct port (Pillow instead of `tf.image`). |
| `resnet_vae.py` | `rdsandwich/models/upper_bound/resnet_vae.py`, `train/train_ub.py --model resnet_vae` | **Full port.** A hierarchical ResNet-VAE with GDN residual encoder/decoder blocks (Cheng et al. 2020), true **bidirectional inference** at each `LatentBlock` (bottom-up + top-down posterior, optionally parameterized relative to the prior), a **channel-wise autoregressive (IAF) prior** on the bottom `--ar_prior_levels` generative levels, and a per-channel **deep factorized** prior on the top latent `z0`. This is the `dim(Z) ≈ 0.66 dim(X)` model of Sec. 6.4. Diagram: `docs/figures/resnet_vae_ladder.png`. |
| `ms2020.py` (β-VAE variant) | `rdsandwich/models/upper_bound/ms2020_vae.py`, `train/train_ub.py --model ms2020_vae` | **Ported architecture, β-VAE recipe.** The Minnen & Singh 2020 channel-autoregressive autoencoder (analysis/synthesis/hyper transforms per Ballé 2018 Table 1, 10 channel slices with LRP), converted to an R-D **upper-bound** β-VAE per the paper's App. A.5.7: factorized-Gaussian posteriors with learned means/variances, a deep factorized hyperprior **not** convolved with a uniform, and a Gaussian channel-conditional prior. The authors' public repo ships only the *operational* `ms2020.py` (rounding/entropy-coded, used as a baseline); the β-VAE upper-bound variant (`rdub-model=ms2020_vae`) had no released code, so this is reconstructed from the architecture + paper. Diagram: `docs/figures/ms2020_vae_diagram.png`. |
| `mbt2018.py`, `ms2020.py` (operational baselines), `biggan.py` | — | **Not ported.** The operational Q-R baseline curves (Minnen 2018/2020, VTM, BPG, JPEG2000, …) in Fig. 3 are published numbers and are not retrained here. The GAN-image experiments (Sec. 6.3, BigGAN source) are out of scope. |
| `boilerplate.py` | — | Keras-training-loop plumbing (LR schedules, callbacks); superseded by `rdsandwich/utils/trainer.py`'s `BaseTrainer` and the per-bound trainer subclasses. |
| `utils.py` | `rdsandwich/utils/` (`io.py`, `torch_utils.py`) | jsonl logging (`get_json_logging_callback` -> `JsonlLogger`), `config_dict_to_str`, checkpoint helpers replacing `model.save_weights`/`load_weights`; re-exported from `rdsandwich.utils`. |

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
