# Templated configs

Fully-documented reference templates for the scripts driven by
[`scripts/run_sweep.py`](../../scripts/run_sweep.py). Each template lists **every
flag** its script accepts, with `(type, default)` and a one-line description, so
you can see what is available without reading the argparse source.
`tests/test_configs.py` parses every template and config with the real CLI
parsers, so they cannot drift out of date.

These are references, not experiments: copy one, trim the flags you don't need,
and move whatever you want to vary into the `sweep:` block. The concrete
experiment configs live one directory up in [`configs/`](..).

## Templates

| Template | Script(s) | Purpose |
|---|---|---|
| [`train_ub.template.yaml`](train_ub.template.yaml) | `train/train_ub.py` | Train an R-D **upper** bound model (`mlp_vae`, `resnet_vae`, `ms2020_vae`) |
| [`eval_ub.template.yaml`](eval_ub.template.yaml) | `evaluation/eval_ub.py` | Evaluate it (full images, or sampled (D, R) with a 95% CI) |
| [`lb.template.yaml`](lb.template.yaml) | `train/train_lb.py` then `evaluation/eval_lb.py` | Train an R-D **lower** bound log-u network, then estimate R_L(D) |

## Which template backs which existing experiment

| Existing config | Template to consult |
|---|---|
| `gaussian_ub.yaml`, `gaussian_ub_zy.yaml`, `physics_ub.yaml` | `train_ub.template.yaml` (`model: mlp_vae`) |
| `natural_images_*_train*.yaml`, `natural_images_ms2020_vae_timing.yaml`, `smoke_natural_images.yaml` | `train_ub.template.yaml` (`model: resnet_vae` / `ms2020_vae`) |
| `natural_images_*_eval.yaml` | `eval_ub.template.yaml` |
| `gaussian_lb.yaml`, `physics_lb.yaml` | `lb.template.yaml` |

## Usage

```bash
python scripts/run_sweep.py --config configs/templates/train_ub.template.yaml --dry-run
```

`--dry-run` prints the expanded commands without running them; drop it to launch,
and add `-j N` to run up to N concurrently.

## Config format (recap)

See [`rdsandwich/sweep.py`](../../rdsandwich/sweep.py):

- `script:` which CLI to run: `train_ub`, `eval_ub`, `train_lb` or `eval_lb`.
  A list (e.g. `[train_lb, eval_lb]`) runs the scripts in order for every combination.
- `fixed:` flags identical across every run (passed to every script in the list).
- `sweep:` flags to vary; `run_sweep.py` runs the Cartesian product of the lists.
- `script_args:` (optional) extra flags for one script only, e.g.
  `script_args: {eval_lb: {num_Ck_samples: 5}}`.

Value to argv conventions:

- `true` -> bare `--flag` (store_true); `false` / `null` -> omitted.
- `[a, b, c]` -> `--flag a,b,c`.
- scalar -> `--flag <value>`.
- **YAML convention:** write learning rates as `5.0e-4`, not `5e-4` (the latter parses as a string).
