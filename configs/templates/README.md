# Templated configs

Fully-documented reference templates for every sweep script driven by
[`scripts/run_sweep.py`](../../scripts/run_sweep.py). Each template lists **every
flag** its script accepts, with `(type, default)` and a one-line description, so
you can see what is available without reading the argparse source.

These are references, not experiments — copy one, trim the flags you don't need,
and move whatever you want to vary into the `sweep:` block. The concrete
experiment configs live one directory up in [`configs/`](..).

## Templates (one per runnable script)

| Template | Script | Purpose |
|---|---|---|
| [`train_rdub.template.yaml`](train_rdub.template.yaml) | `train/train_rdub.py` | R-D **upper** bound (β-VAE) |
| [`train_rdlb.template.yaml`](train_rdlb.template.yaml) | `train/train_rdlb.py` | R-D **lower** bound (log-u network) |
| [`train_resnet_vae.template.yaml`](train_resnet_vae.template.yaml) | `train/train_resnet_vae.py` | Image UB, minimal ResNet-VAE trainer (GAN-image / Sec. 6.3) |
| [`train_image_ub.template.yaml`](train_image_ub.template.yaml) | `train/train_image_ub.py` | Natural-image UB, full pipeline (train+eval, resnet_vae + ms2020_vae) |

## Which template backs which existing experiment

| Existing config | Template to consult |
|---|---|
| `gaussian_ub.yaml`, `gaussian_ub_zy.yaml`, `physics_ub.yaml` | `train_rdub.template.yaml` |
| `gaussian_lb.yaml`, `physics_lb.yaml` | `train_rdlb.template.yaml` |
| `natural_images_resnet_vae_train.yaml`, `natural_images_resnet_vae_eval.yaml` | `train_image_ub.template.yaml` (`model: resnet_vae`) |
| `natural_images_ms2020_vae_train.yaml`, `natural_images_ms2020_vae_eval.yaml` | `train_image_ub.template.yaml` (`model: ms2020_vae`) |
| `smoke_natural_images.yaml` | `train_image_ub.template.yaml` |
| (GAN basenji d=2/d=4, Sec. 6.3) | `train_resnet_vae.template.yaml` |

## Usage

```bash
python scripts/run_sweep.py --config configs/templated/train_rdub.template.yaml --dry-run
```

`--dry-run` prints the expanded commands without running them; drop it to launch,
and add `-j N` to run up to N concurrently.

## Config format (recap)

A config has three top-level keys (see [`rdsandwich/config.py`](../../rdsandwich/config.py)):

- `script:` — which `train/<script>.py` to run.
- `fixed:` — flags identical across every run.
- `sweep:` — flags to vary; `run_sweep.py` runs the Cartesian product of the lists.

Value → argv conventions:

- `true` → bare `--flag` (store_true); `false` / `null` → omitted.
- `[a, b, c]` → `--flag a,b,c`.
- scalar → `--flag <value>`.
- **YAML convention:** write learning rates as `5.0e-4`, not `5e-4` (the latter parses as a string).
