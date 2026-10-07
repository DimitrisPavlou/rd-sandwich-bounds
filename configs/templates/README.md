# Templated configs

Fully-documented reference templates for the scripts driven by
[`scripts/run_sweep.py`](../../scripts/run_sweep.py), grouped by data source like
the experiment configs one directory up:

```
configs/
  gaussian/    physics/    images/          experiment configs
  templates/
    gaussian/  physics/    images/          one template per (model, script)
```

File names say only the model, the script and the bound, since the folder says the
source: `<model>_<train|eval>_<ub|lb>[_<variant>].yaml`, with `train_eval` for the
lower-bound configs (they train, then evaluate) and `.template.yaml` for templates.
For example `configs/images/resnet_vae_train_ub_pm1.yaml` or
`configs/templates/physics/mlp_train_eval_lb.template.yaml`.

Each template lists **every flag** its script accepts for that model, with
`(type, default)` and a one-line description, so you can see what is available
without reading the argparse source. Flags left at their default are shown
commented out. `tests/test_configs.py` parses every template and config with the
real CLI parsers, checks that each template lists every flag of its model, and
checks the folder / file-name scheme, so they cannot drift out of date.

These are references, not experiments: copy one into `configs/<source>/`, trim the
flags you don't need, and move whatever you want to vary into the `sweep:` block.

## Layout of every template and config

Inside `fixed:`, flags are grouped in three sections, in this order:

1. **data**: the dataset and how it is loaded (paths, crops, h-flip, padding, workers);
2. **model**: the model and its hyperparameters (including `lambda` / `lamb`);
3. **training** (or **evaluation**, or **training and evaluation** for the lower
   bound): batch size, epochs, optimizer, LR schedule, EMA, checkpoints, run bookkeeping.

`sweep:` comes after `fixed:`; each swept key is commented with the section it belongs to.

## Templates

| Template | Script(s) | Model |
|---|---|---|
| [`gaussian/mlp_vae_train_ub.template.yaml`](gaussian/mlp_vae_train_ub.template.yaml) | `train/train_ub.py` | MLP beta-VAE, n=1000 Gaussian |
| [`gaussian/mlp_vae_eval_ub.template.yaml`](gaussian/mlp_vae_eval_ub.template.yaml) | `evaluation/eval_ub.py` | MLP beta-VAE ((D, R) with a 95% CI) |
| [`gaussian/mlp_train_eval_lb.template.yaml`](gaussian/mlp_train_eval_lb.template.yaml) | `train/train_lb.py` then `evaluation/eval_lb.py` | lower bound, MLP log-u network |
| [`physics/mlp_vae_train_ub.template.yaml`](physics/mlp_vae_train_ub.template.yaml) | `train/train_ub.py` | MLP beta-VAE with a MAF prior, particle physics |
| [`physics/mlp_vae_eval_ub.template.yaml`](physics/mlp_vae_eval_ub.template.yaml) | `evaluation/eval_ub.py` | MLP beta-VAE ((D, R) with a 95% CI) |
| [`physics/mlp_train_eval_lb.template.yaml`](physics/mlp_train_eval_lb.template.yaml) | `train/train_lb.py` then `evaluation/eval_lb.py` | lower bound, MLP log-u network |
| [`images/resnet_vae_train_ub.template.yaml`](images/resnet_vae_train_ub.template.yaml) | `train/train_ub.py` | hierarchical ResNet-VAE (Yang & Mandt); values: the shared recipe |
| [`images/resnet_vae_eval_ub.template.yaml`](images/resnet_vae_eval_ub.template.yaml) | `evaluation/eval_ub.py` | ResNet-VAE (full images: per-image bpp / MSE / PSNR) |
| [`images/variable_rate_lossy_vae_train_ub.template.yaml`](images/variable_rate_lossy_vae_train_ub.template.yaml) | `train/train_ub.py` | variable-rate ResNet-VAE (Duan et al. 2023); values: the shared recipe |
| [`images/variable_rate_lossy_vae_eval_ub.template.yaml`](images/variable_rate_lossy_vae_eval_ub.template.yaml) | `evaluation/eval_ub.py` | variable-rate model, one checkpoint at many lambdas |
| [`images/ms2020_vae_train_ub.template.yaml`](images/ms2020_vae_train_ub.template.yaml) | `train/train_ub.py` | Minnen & Singh 2020 beta-VAE |
| [`images/ms2020_vae_eval_ub.template.yaml`](images/ms2020_vae_eval_ub.template.yaml) | `evaluation/eval_ub.py` | Minnen & Singh 2020 beta-VAE |
| [`images/cnn_train_eval_lb.template.yaml`](images/cnn_train_eval_lb.template.yaml) | `train/train_lb.py` then `evaluation/eval_lb.py` | lower bound, CNN log-u network |

## Which template backs which experiment config

| Experiment config (under `configs/`) | Template (under `configs/templates/`) |
|---|---|
| `gaussian/mlp_vae_train_ub.yaml`, `gaussian/mlp_vae_train_ub_zy.yaml` | `gaussian/mlp_vae_train_ub.template.yaml` |
| `gaussian/mlp_train_eval_lb.yaml` | `gaussian/mlp_train_eval_lb.template.yaml` |
| `physics/mlp_vae_train_ub.yaml` | `physics/mlp_vae_train_ub.template.yaml` |
| `physics/mlp_train_eval_lb.yaml` | `physics/mlp_train_eval_lb.template.yaml` |
| `images/resnet_vae_train_ub.yaml`, `..._light.yaml`, `..._pm1.yaml` | `images/resnet_vae_train_ub.template.yaml` |
| `images/resnet_vae_eval_ub.yaml`, `images/resnet_vae_eval_ub_pm1.yaml` | `images/resnet_vae_eval_ub.template.yaml` |
| `images/variable_rate_lossy_vae_train_ub.yaml`, `..._check.yaml` | `images/variable_rate_lossy_vae_train_ub.template.yaml` |
| `images/variable_rate_lossy_vae_eval_ub.yaml` | `images/variable_rate_lossy_vae_eval_ub.template.yaml` |
| `images/ms2020_vae_train_ub.yaml`, `images/ms2020_vae_train_ub_timing.yaml` | `images/ms2020_vae_train_ub.template.yaml` |
| `images/ms2020_vae_eval_ub.yaml` | `images/ms2020_vae_eval_ub.template.yaml` |
| `images/smoke_train_ub.yaml` (both models, tiny) | `images/resnet_vae_train_ub` + `images/ms2020_vae_train_ub` templates |

## Usage

```bash
python scripts/run_sweep.py --config configs/templates/images/resnet_vae_train_ub.template.yaml --dry-run
```

`--dry-run` prints the expanded commands without running them; drop it to launch,
add `-j N` to run up to N concurrently, or `--index N` to run only the N-th one
(e.g. a Slurm array task).

## Config format (recap)

See [`rdsandwich/sweep.py`](../../rdsandwich/sweep.py):

- `script:` which CLI to run: `train_ub`, `eval_ub`, `train_lb` or `eval_lb`.
  A list (e.g. `[train_lb, eval_lb]`) runs the scripts in order for every combination.
- `fixed:` flags identical across every run (passed to every script in the list),
  in the three sections above.
- `sweep:` flags to vary; `run_sweep.py` runs the Cartesian product of the lists.
- `script_args:` (optional) extra flags for one script only, e.g.
  `script_args: {eval_lb: {num_Ck_samples: 5}}`.

Value to argv conventions:

- `true` -> bare `--flag` (store_true); `false` / `null` -> omitted.
- `[a, b, c]` -> `--flag a,b,c`.
- scalar -> `--flag <value>`.
- **YAML conventions:** write learning rates as `5.0e-4`, not `5e-4` (the latter
  parses as a string); quote `"0_255"` (unquoted, YAML reads it as an octal number).
