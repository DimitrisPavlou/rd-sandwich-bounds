# Plan: unified retraining of the image R-D upper-bound models

Status: Steps 1–5 done, with tests 6.1–6.6 passing (2026-10-07). Step 7 (the runs) not started.

## Goal

Retrain the image R-D upper-bound models under **one shared recipe**, so that their
(bpp, PSNR) curves differ only because of the model:

- `resnet_vae`: Yang & Mandt, ICLR 2022 (existing, gets a `[-1, 1]` mode).
- `variable_rate_lossy_vae`: new port of Duan, Ma, He & Zhu, *An Improved Upper Bound on the
  Rate-Distortion Function of Images*, ICIP 2023
  (code: `lvae/models/rd` in https://github.com/duanzhiihao/lossy-vae).

Out of scope for now: `ms2020_vae` (can be added later under the same recipe), BD-rate,
VTM anchors and VTM curves, loading Duan's pretrained checkpoint.

## Decisions

| # | Decision |
|---|---|
| 1 | Models: `resnet_vae` + `variable_rate_lossy_vae` only (no `ms2020_vae` for now). |
| 2 | Training data: raw COCO train2017 (all 118,287 images), as in Duan et al. |
| 3 | Duan model size: `base` (C=128, N=15, 186.7M params) for the real run; `c64_l5` (34.2M) only as a pipeline check. |
| 4 | Batch size is a hyperparameter (`--batchsize`, recipe value 32). No gradient accumulation. |
| 5 | Fixed-rate λ grid: λ_d ∈ {8, 22.6, 64, 181, 512, 1448}. |
| 6 | `variable_rate_lossy_vae` is trained variable-rate only (no fixed-λ control runs). |
| 7 | Evaluation does not round x̂ to 8 bits (matches Duan); the rounded PSNR is also stored from the same forward pass (see Step 3). |
| 8 | No pretrained download; the port is verified against Duan's source code instead (Step 6.2). |
| 9 | Model name: `variable_rate_lossy_vae`. |

Also agreed:

- **One loss for every model, Duan's form:**
  `loss = KL_nats / (3·H·W) + λ_d · MSE_[-1,1]`,
  so a given λ_d means the same thing for every model.
  Conversion from the old `resnet_vae` convention (`bpp + λ·MSE_[0,255]`): `λ_d ≈ 3756 · λ_old`
  (exact factor 255² / (4 · 3 · log2 e)).
- **Reconstruction target in [-1, 1].** Input normalisation stays model-specific:
  `resnet_vae` keeps `x/255 − 0.5` (GDN starts in the same regime as before);
  `variable_rate_lossy_vae` keeps its ImageNet normalisation (shift −0.4546, scale 3.6757).
- **Reporting is always bpp and PSNR on the [0, 255] scale** (output contract, Step 3).

## Background: what Duan et al. do

- Bound (paper eqs. 3–6): `I(X; X̂) ≤ E[Σ_i KL(q(z_i | x, z_<i) ‖ p(z_i | z_<i))]`;
  train on `Σ KL + λ·MSE`, then measure D = E[MSE] and U(D) = E[Σ KL] on the test set.
- Three changes over Yang & Mandt: a larger ResNet VAE (Table 1: channels 2C/4C/5C/6C/6C at
  16×16…1×1, latents per scale 5/4/3/2/1), the smoothing function
  `μ = sign(a)·|a|^(1 − 0.5·tanh|a|)` on posterior and prior means (eq. 8), and variable-rate
  training with λ log-uniform in [4, 2048].
- Reported: −31.4% (Kodak), −34.5% (Tecnick), −31.9% (CLIC) BD-rate vs VTM 18.0; Yang & Mandt
  −19.3% (Kodak).
- Paper vs code differences, and which we follow:

| Item | Paper | Code | We follow |
|---|---|---|---|
| λ range | "U(log 2, log 2048)" | log 4 … log 2048 | code (the paper text says [4, 2048]) |
| Batch size | 32 | README command uses 16 | hyperparameter, recipe value 32 |
| Loss normalisation | eq. 5 unnormalised | KL nats/(3HW) + λ·MSE on [-1, 1] | code |
| Smoothing | eq. 8 | `linear_sqrt` (signed √ for \|a\| > 6) | code (numerically the same) |
| σ parameterisation | not given | softplus with β = ln 2 | code |
| bpp at eval | bits / pixels | bits / padded pixels | ours: bits / original pixels |

## Step 1: port `variable_rate_lossy_vae`

**`rdsandwich/layers/convnext_adaln.py`** (from `lvae/models/common.py` and `lvae/models/rd/model.py`):

- `get_conv`, `conv_k1s1`, `conv_k3s1`, `patch_downsample`, `patch_upsample`,
  `sinusoidal_embedding`, `ConvNeXtBlockAdaLN`, `ConvNeXtAdaLNPatchDown`.
- A local `Mlp` (`fc1`, `act`, `fc2`) instead of `timm.layers.Mlp`, so there is no timm dependency.
- Keep Duan's attribute names (including the typo `downsapmle`) so state dicts match theirs.

**`rdsandwich/models/upper_bound/variable_rate_lossy_vae.py`:**

- `VariableRateLossyVAEConfig`: `base_channels` (C), `latents_per_scale` (ordered 1×1 → 16×16),
  `z_dim = 32`, `smooth = True`, `lmb_range = (4, 2048)`, `lmb_embed_dim = (256, 256)`,
  `sin_period = 64`.
- Presets (Tables 1 and 3):

| Preset | C | latents per scale (1×1 → 16×16) | Params |
|---|---|---|---|
| `base` | 128 | [1, 2, 3, 4, 5] | 186.7M |
| `c96_l15` | 96 | [1, 2, 3, 4, 5] | 111.8M |
| `c64_l15` | 64 | [1, 2, 3, 4, 5] | 55.8M |
| `c64_l10` | 64 | [1, 2, 2, 2, 3] | 46.1M |
| `c64_l5` | 64 | [1, 1, 1, 1, 1] | 34.2M |
| `c32_l15` | 32 | [1, 2, 3, 4, 5] | 18.6M (not in the paper; about the ResNet-VAE's 15.7M) |
| `tiny` | small | small | tests / smoke only |

  `smooth=False` gives the paper's no-smoothing ablation (Duan's `LatentVariableBlockOld`).
- Port: `LatentVariableBlock` (with `linear_sqrt`, softplus-β=ln 2 for σ), `FeatureExtractor`,
  λ embedding (log → sinusoidal → MLP), log-uniform λ sampling per image, the learned `bias`.
  Reuse `rdsandwich.models.upper_bound._common.gaussian_kl` (same formula).
- Do not port: sampling / `study` / `_self_evaluate` / FLOPs mode, wandb, DDP, `zoo_ablation.py`.
- Interface: `get_losses(x)` and `forward(x, lmbda=None)` per the output contract (Step 3).
  Training samples λ_d per image; evaluation uses the given `lmbda`.
- `diagnostics(x)`: per-latent q/p statistics for the trainer's explosion report.

**CLI (`rdsandwich/cli/upper_bound.py`):** add `variable_rate_lossy_vae` to `MODELS`; flags `--preset`,
`--base_channels`, `--latents_per_scale`, `--no_smooth`, `--lmb_range`;
`downsampling_factor = 64`; run name without λ, built from the resolved architecture, e.g.
`rdub-model=variable_rate_lossy_vae-C=128-L=1_2_3_4_5-lmb=4_2048` (`-nosmooth` / `-tiny` appended when set).
At evaluation `--lambda` is λ_d and the same checkpoint serves every λ; a λ outside `lmb_range`
(e.g. an old [0, 255]-convention value) raises an error.

Done: `rdsandwich/layers/convnext_adaln.py`, `rdsandwich/models/upper_bound/variable_rate_lossy_vae.py`,
CLI registration, `tests/test_variable_rate_lossy_vae.py`. `tensor_stats` moved from `resnet_vae.py` to
`_common.py` (shared by both models).

## Step 2: `[-1, 1]` mode for `resnet_vae`

- New config field `ResNetVAEConfig.image_range: "0_255" | "pm1"`, default `"0_255"` so the
  existing checkpoints still load and evaluate. The shared configs set `"pm1"`.
- With `pm1`:
  - output head emits [-1, 1] directly (drop `+0.5, ×255` at `resnet_vae.py:119`);
  - target is `x/127.5 − 1`; the input stays `x/255 − 0.5` (`resnet_vae.py:98`);
  - loss is `bits·ln2/(3HW) + λ_d · MSE_[-1,1]`;
  - `--lambda` means λ_d;
  - the run name gains `-range=pm1`.
- With λ converted, `pm1` reaches the same optimum as `0_255`: the head is the old head with its
  last conv doubled, and the loss is 1/(3·log2 e) ≈ 0.231 × the old loss.

Done: `ResNetVAEConfig.image_range`, `--image_range {0_255,pm1}` (warns when a `pm1` λ looks like
an old-convention value, < 1), run name `rdub-model=resnet_vae-range=pm1-lambda=…`,
`MSE_PM1_TO_255` shared in `_common.py`, tests in `tests/test_resnet_vae.py`. The existing
`0_255` checkpoints load and evaluate unchanged. Left for Step 3: `eval_ub.py` still names its
output `rdub-model=resnet_vae-lambda=…` for `pm1` runs (no range tag).

## Step 3: output contract (bpp and PSNR for every model)

Every image model must satisfy:

- `forward(x)`: `x` in [0, 255] → `x_hat` in [0, 255] (converted at the boundary) and per-image
  `bits` (in bits, shape [B]).
- `get_losses(x)`: returns `(loss, bpp, mse_255)`. Only `loss` uses the model's own units.

Evaluation and plots:

- `evaluate_full_images`: clamp x̂ to [0, 255]; bpp = bits / original pixels; PSNR per image,
  averaged over images. Stores `psnr`/`mse` (unrounded, used by default) **and**
  `psnr_uint8`/`mse_uint8` (x̂ rounded to 8 bits), both from the same forward pass.
- Eval file name `rdub-model=<key>-lambda=<λ_d>-dataset=<ds>.npz`, where `<key>` includes the
  tags (e.g. `resnet_vae-range=pm1`), so old and new curves never merge in `plot_qr.py`.
- `plot_qr.py`: legend labels for the new keys; `--uint8` flag to plot the rounded PSNR.
- `plot_image_ub_training.py`: run-name regex accepts `-range=pm1`; skip variable-rate runs
  (their training logs average over random λ).

Done:
- `evaluate_full_images` always stores `bpp`, `mse`, `psnr` (unrounded) and `mse_uint8`,
  `psnr_uint8` from the same forward pass; `--no_cast_xhat` removed (both are always stored).
- `model_key(args)` in `rdsandwich/cli/upper_bound.py` names a curve (`resnet_vae`,
  `resnet_vae-range=pm1`, `variable_rate_lossy_vae-C=128-L=1_2_3_4_5-lmb=4_2048`, ...); used by run names and
  by `eval_ub.py`'s output file names.
- `plot_qr.py`: plots every key found in `--results_dir` by default (`--models` to restrict),
  `--uint8` for the rounded distortion, old single-distortion files plotted with a note, the
  first curve of each model in that model's colour.
- `plot_image_ub_training.py`: regex accepts `-range=…` and exponent λ values (`1e-05`).
- Config templates document `--image_range` (quote `"0_255"` in YAML: unquoted it is an octal
  number) and the `variable_rate_lossy_vae` flags.
- Tests: `tests/test_image_model_contract.py` (all image models, both ranges),
  `tests/test_result_naming_and_plots.py`, evaluator rounding/clipping in `tests/test_upper_bound.py`.

## Step 4: trainer and data for the shared recipe

All opt-in; existing commands behave as before.

- Budget stays in epochs (`--epochs`), as today: one epoch = one pass over the training images,
  `steps per epoch = ⌊num_images / batchsize⌋` (the last partial batch is dropped), and
  `total steps ≈ epochs × steps per epoch`. No iteration counting is added. Logging, checkpoints
  (`--checkpoint_interval`) and resume stay per epoch.
  Example (shared recipe): COCO, 118,287 images, batch 32 → 3,696 steps/epoch; 54 epochs →
  199,584 steps ≈ Duan's 200k. Because the budget is in epochs, a different batch size keeps the
  number of training patches fixed and changes the number of steps (batch 16 → 399,168 steps).
- `--lr_schedule const-cos`: constant for the first half of the epochs, then a cosine down to
  0.01× over the second half; stepped once per epoch like the existing schedules; state saved in
  the checkpoint.
- `--ema 0.9999`, with Duan's warmup `d·(1 − e^(−step/10000))`, where `step` is the global step
  count (`epoch × steps per epoch + step within the epoch`); EMA weights saved in the checkpoint
  and used by `eval_ub.py` by default (`--no_ema` for the raw weights).
- `ImageFolderDataset`:
  - `--hflip`: mirror each training patch left–right with probability 0.5.
  - `--small_image_mode {resize,reflect_pad}` (default `resize` = today's behaviour) for images
    smaller than the crop. `reflect_pad` does what Duan's
    `RandomCrop(256, pad_if_needed=True, padding_mode='reflect')` does on PIL images: pad each
    too-short side on both ends by the shortfall (256 − side) with a mirror image of the content
    (`numpy.pad(mode="reflect")`, which works for any image size), then crop randomly.
    - Why: a random 256×256 patch needs the image to be at least 256 px in both directions; some
      raw COCO images are smaller in one direction (e.g. 640×200). Today's `resize` *stretches*
      such an image up, inventing smoother-than-real pixels by interpolation. `reflect_pad`
      *mirrors* the image at its edges instead, so every pixel is real photo content.
    - Example: a 200-px-tall image gets 56 rows above (mirror of its top 56 rows) and 56 rows below
      (mirror of its bottom 56 rows), becoming 312 rows tall; the 256-row patch is cropped from it
      at a random position.
    - Any size works: PyTorch's reflect padding cannot add more pixels than the image has (a
      100-px side would need 156 on each end), but `numpy.pad(mode="reflect")`, which torchvision
      uses for PIL images and therefore Duan's code too, keeps reflecting back and forth as long
      as needed. So there is no fallback to stretching and nothing to count.
    - Only matters for raw COCO: the current 8,442-image set has no side under ~307 px
      (`prepare_imgs.py` kept images ≥ 512 px and downsampled them by at most 0.6×).
- Full COCO uses DataLoader workers, not `--preload` (≈100 GB of uint8).
- Resume (per epoch, as today) also restores the EMA weights and the global step count, so 48 h
  Slurm jobs chain.
- Not ported: Duan's "cut the LR 10× on a gradient spike".
- Checkpoint file name: `train_ub.py` writes `ckpt-lambda=<λ>.pt`; for the variable-rate model
  the λ there is meaningless, so name it `ckpt.pt` for `variable_rate_lossy_vae`.

Done:
- `--lr_schedule const-cos` (`const_cos_factor` / `make_const_cos_scheduler` in
  `rdsandwich/utils/lr_schedulers.py`, with the piecewise schedule).
- `--ema DECAY --ema_warmup STEPS` in `BaseTrainer` (any trainer): EMA updated after every
  optimizer step, `global_step` counted across epochs and resumes; both saved in the checkpoint
  (`extra["ema_state_dict"]`, `extra["global_step"]`) and restored by `--resume`.
  `load_checkpoint(..., use_ema=True)` loads the EMA weights; `eval_ub.py` does this by default
  and prints which weights it used (`--no_ema` for the raw ones).
- `--hflip` and `--small_image_mode {resize,reflect_pad}` (`rdsandwich/data/image.py`, passed
  through `get_dataset` / `load_dataset`); `all_images()` (evaluation) is never augmented.
- `checkpoint_filename(args)`: `ckpt.pt` for `variable_rate_lossy_vae`, `ckpt-lambda=<λ>.pt` otherwise.
- `train_ub.py` prints `steps/epoch` and the total steps for the run.
- Config templates document the new flags; tests in `tests/test_training_recipe.py`
  (reflect padding is checked to match torchvision's `RandomCrop` padding exactly, including
  sides of 1 and 70 px).
- Notes: the `lr` field in the jsonl log is the LR *after* that epoch's scheduler step, i.e. the
  one the next epoch uses (existing behaviour). `--resume` assumes the same `--epochs` as the
  interrupted run; changing it mid-run reshapes the `const-cos` schedule.

## Step 5: configs and scripts

Done. Layout: configs grouped by data source, `configs/{gaussian,physics,images}/`, and
templates mirroring it, `configs/templates/{gaussian,physics,images}/`; names say only
`<model>_<train|eval|train_eval>_<ub|lb>[_<variant>]` (`.template.yaml` for templates).
`.gitignore` keeps experiment configs local (`configs/*/*.yaml`; the already-tracked gaussian and
physics configs stay tracked) and versions the templates.

- Every template and config groups its `fixed:` flags in three sections, in this order:
  1. data, 2. model, 3. training (or evaluation; for the lower bound, training and evaluation).
- Templates, one per (source, model, script), each listing every flag that model accepts with
  `(type, default)` and a short comment (replacing the old all-in-one templates):
  `gaussian/` and `physics/`: `mlp_vae_{train,eval}_ub`, `mlp_train_eval_lb`;
  `images/`: `{resnet_vae,variable_rate_lossy_vae,ms2020_vae}_{train,eval}_ub`, `cnn_train_eval_lb`.
  `tests/test_configs.py` checks that each lists every flag of its model, has the sections,
  and that every config follows the folder / name scheme.
- The 12 existing configs: reordered into the three sections; every value unchanged
  (checked: each expands to exactly the same commands as before).
- New experiment configs (the shared recipe below):
  - `images/resnet_vae_train_ub_pm1.yaml` / `images/resnet_vae_eval_ub_pm1.yaml` (6 λ_d; eval × Kodak, Tecnick);
  - `images/variable_rate_lossy_vae_train_ub.yaml` / `images/variable_rate_lossy_vae_eval_ub.yaml`
    (1 run; eval at 22 λ_d × 2 sets);
  - `images/variable_rate_lossy_vae_train_ub_check.yaml` (`c64_l5`, 2 epochs: the pipeline check).
  They expect `data/coco_train2017` → raw COCO train2017 (a symlink).
- `scripts/run_sweep.py --index N` runs only the N-th run of a config;
  `scripts/slurm/train_ub_unified.sbatch` runs array task i as `--index i` of `$CONFIG`.
- `scripts/smoke/smoke_unified.sh`: both models, tiny, all recipe flags, train → resume → eval
  (EMA) → both plots; passes on raw COCO, also with `COMPILE=1`.
- Fixed in passing: the smoke scripts imported `dataset_name` from `rdsandwich.data`, which no
  longer exports it; they now import it from `rdsandwich.data.datasets`.
- README: new model, flags, output contract, the shared-recipe section, `--index`, config layout.
- Measured GPU memory, one fp32 training step at 256×256: Duan `c64_l5` 0.4 GiB/patch
  (batch 16: 6.5 GiB), Duan `base` 8.4 GiB at batch 4 (batch 32 ≈ 45 GiB by extrapolation),
  ResNet-VAE 5.6 GiB at batch 16. Batch 32 needs the cluster for all of them (80 GB H100 for
  Duan `base`).
- `compile: true` is set for the ResNet-VAE (as in the old runs) and not for Duan's model;
  compile works on the tiny Duan model in the smoke test, so it can be turned on after a check
  on the base model.

## Shared recipe

| Item | Setting |
|---|---|
| Training data | raw COCO train2017, 118,287 images; random 256×256 crop (`--small_image_mode reflect_pad`) + `--hflip` |
| Budget | 54 epochs, identical for every model and every λ. At batch 32 on COCO: ⌊118,287 / 32⌋ = 3,696 steps/epoch → 54 × 3,696 = 199,584 steps (≈ Duan's 200k; 6.39M patches). Changing the batch size keeps the patches seen fixed and changes the step count (batch 16: 399,168 steps) |
| Optimiser | Adam, lr 2e-4, no weight decay |
| LR schedule | `const-cos` |
| Grad clip | 2.0 (global norm, nats/dim loss units) |
| EMA | 0.9999 |
| Precision | fp32 |
| Loss | `KL_nats / (3·H·W) + λ_d · MSE_[-1,1]` |
| Checkpoint | final EMA weights only; no selection on any test set |
| Divergence rule | only that model's lr may be lowered, and it is recorded |

Fixed-rate λ grid: λ_d ∈ {8, 22.6, 64, 181, 512, 1448}.
`variable_rate_lossy_vae` is evaluated at the same six λ_d plus 16 λ_d log-spaced over [4, 2048].

Evaluation: Kodak and Tecnick (`RGB_OR_1200x1200`), whole images, reflect-padded to a multiple of
the model's downsampling (64), fixed seed, unrounded x̂ by default.

Caveat: the old `resnet_vae` runs used `grad_clip 5000` in bpp units (≈1155 in nats/dim units);
with 2.0, clipping will likely be active on most steps for `resnet_vae`. Check `clip_frac` and
the gradient norm in the smoke run; if needed, use a per-model clip value and record it.

## Step 6: verification (before any long run)

1. Unit tests for `variable_rate_lossy_vae` (`tests/test_variable_rate_lossy_vae.py`): shapes on `tiny`, finite loss
   and backward, λ embedding, bounded gradient of the smoothing function, state-dict key names
   match Duan's, parameter counts match Table 3 (34.2M / 46.1M / 55.8M / 111.8M / 186.7M).
2. Numerical equivalence with Duan's code (one-off dev check, not a repo test): build their model
   from a local clone of `lossy-vae` (stubbing `timm.layers.mlp.Mlp`), copy its random weights
   into ours, and check that both give the same KL and x̂ for the same input, λ and noise.
   Result: all 7 of their zoo models (`rd_model_base` + 6 ablations) load strictly into the
   matching preset, parameter counts are identical, and KL, x̂, loss and bpp agree to ≤ 2e-7
   relative at λ ∈ {4, 64, 2048}. (Their `zoo_ablation.py` refers to `lib.ConvNeXtBlockAdaLN`,
   which only exists in `lvae/models/common.py`; the check patches that name.)
3. `pm1` equivalence test: with the head's last conv doubled and λ_d = 3756·λ_old, the `pm1` loss
   equals 0.231 × the `0_255` loss, and the logged bpp and MSE_255 are identical.
4. Output-contract test for every image model and both ranges.
5. Trainer and data tests: EMA update and warmup, `const-cos` LR per epoch, resume restores the
   EMA and the step count, `--hflip`, `reflect_pad` on images smaller than the crop (including
   sides under 128 px).
6. Smoke script: all models → eval → both plots.

## Step 7: experiment runs

| Run | Count | Notes |
|---|---|---|
| `variable_rate_lossy_vae`, `c64_l5`, variable-rate | 1 | pipeline check only (short) |
| `variable_rate_lossy_vae`, `base`, variable-rate | 1 | likely several GPU-days; chained 48 h jobs |
| `resnet_vae`, `pm1`, fixed λ_d | 6 | architecture `--latent_channels 4,8,16,32,64,128 --ar_prior_levels 4 --ar_slices 8 --num_filters 256` |

Final figures: `plot_qr.py` (PSNR vs bpp, and `--rd`) on Kodak and Tecnick, one curve per model key.

## Build order

Step 1 + tests 6.1–6.2 → Steps 2–3 + tests 6.3–6.4 → Step 4 + test 6.5 → Step 5 + smoke run 6.6 →
long runs (Step 7).
