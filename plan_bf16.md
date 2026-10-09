# Plan: bf16 mixed precision, with fp16 kept as an option

Status: not started (2026-10-09). Nothing below is implemented yet.

## Goal

Make bf16 the default mixed-precision mode for `train/train_ub.py`, keep the current fp16
mode (fp16 autocast + `GradScaler`) selectable for backward compatibility, and keep the
trainer code simple to read.

Why: under `--amp` (fp16) every epoch logs 1-3 `nonfinite_grad_steps`. These are the
`GradScaler` probing a larger loss scale every 2000 clean steps and overflowing fp16's
maximum (65,504); the step is skipped and the scale halved. bf16 has fp32's exponent range,
so it needs no loss scaling: no scaler, no overflow probes, no skipped steps, and no
`.item()` sync inside `scaler.step()`. On A100/H100 bf16 and fp16 run at the same
tensor-core speed. The cost is a coarser mantissa (8 bits vs fp16's 11), see
"Where bf16 still rounds" below.

## Decisions

| # | Decision |
|---|---|
| 1 | One flag, `--amp [bf16\|fp16]`. Bare `--amp` means bf16; no flag means fp32. No second `--amp_dtype` flag (it would allow `--amp_dtype fp16` without `--amp`, which silently does nothing). |
| 2 | Loss scaling (`GradScaler`) only for fp16. |
| 3 | Under bf16, a non-finite gradient is real: it takes the same path as in fp32 (explosion report, then skip up to `--skip_nonfinite` or abort). |
| 4 | The precision is recorded in the checkpoint. Resuming with a different precision warns but does not raise, so a deliberate switch stays possible. |
| 5 | Evaluation stays fp32 (it never used autocast), so reported (D, R) points do not depend on the training precision. |
| 6 | `resnet_vae` and `variable_rate_lossy_vae` always use the same precision, so precision is not a second difference between their curves. |
| 7 | Running the latent-head convs, image head and DeepFactorized matmuls in fp32 is a separate follow-up (Step 8), because it changes numbers. |

| Command line | YAML (`run_sweep.py`) | Result |
|---|---|---|
| no flag | `amp: false` / omitted | fp32 (unchanged) |
| `--amp` | `amp: true` | **bf16** (new default) |
| `--amp bf16` | `amp: bf16` | bf16 |
| `--amp fp16` | `amp: fp16` | fp16 + GradScaler (the previous behaviour) |

`rdsandwich/sweep.py:params_to_argv` already renders `true` as a bare `--amp` and a string
as `--amp fp16`, so existing YAML and bash keep working without changes.

## Step 1: trainer (`rdsandwich/utils/trainer.py`)

One table of autocast dtypes, next to `BaseTrainer`, as the single source of truth (the CLI
imports its keys):

```python
# Autocast dtypes. Only fp16 needs loss scaling: bf16 has fp32's exponent range.
AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
```

`__init__`: the argument `amp: bool = False` becomes `amp: Optional[str] = None`, and the
two lines that set `self.amp` / `self.scaler` become:

```python
if amp is not None and amp not in AMP_DTYPES:
    raise ValueError(f"amp must be one of {sorted(AMP_DTYPES)} or None, got {amp!r}")
use_amp = amp is not None and self.device.type == "cuda"
if use_amp and amp == "bf16" and not torch.cuda.is_bf16_supported():
    raise RuntimeError("bf16 autocast needs an Ampere-or-newer GPU; use --amp fp16.")
self.amp = amp if use_amp else None                 # "bf16" | "fp16" | None, as recorded
self.amp_dtype = AMP_DTYPES[amp] if use_amp else None
self.grad_scaling = self.amp == "fp16"              # GradScaler only for fp16
self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.grad_scaling)
```

Two flags with one meaning each: `self.amp` / `self.amp_dtype` say whether and in what
dtype autocast runs, `self.grad_scaling` says whether the scaler is on.

The rest of the file:

| Where | Change |
|---|---|
| autocast in `train()` | `torch.autocast(..., dtype=self.amp_dtype, enabled=self.amp is not None)` |
| non-finite branch in `train()` (`if self.amp:`) | becomes `if self.grad_scaling:`; comment: only fp16 overflows are routine. **Safety-critical**: with the scaler disabled, `scaler.step()` is a plain `optimizer.step()`, so under bf16 a non-finite gradient must not go down the "the scaler will skip it" branch, or inf/NaN reaches the weights and the Adam state. |
| loss averaging (`0 if self.amp else n_nonfinite`) | `self.amp` becomes `self.grad_scaling` |
| `_save` | always store `extra["amp"] = self.amp`; store `extra["scaler"]` only if `grad_scaling` |
| `_maybe_resume` | load the scaler state only if `grad_scaling`; compare the precision (Step 3) |
| `_explosion_report` | `"amp": self.amp`; `"amp_scale"` only if `grad_scaling` |

`amp_nonfinite_patience` keeps its name (API compatibility); its comment becomes
"consecutive fp16 GradScaler overflows". `LowerBoundTrainer` never sets `amp` and is
unaffected.

## Step 2: CLI (`rdsandwich/cli/upper_bound.py`)

```python
g.add_argument("--amp", nargs="?", const="bf16", default=None, choices=sorted(AMP_DTYPES),
               help="Mixed precision. --amp / --amp bf16: bf16 autocast, no loss scaling. "
                    "--amp fp16: fp16 autocast + GradScaler (the previous behaviour). Off: fp32.")
```

`train/train_ub.py` is unchanged: it already passes `amp=args.amp`, now a string or `None`.
None of the parsers has positional arguments, so `--amp --compile` does not swallow
`--compile`.

## Step 3: resuming older runs

The run directory name depends only on the model, architecture and lambda, not on the
precision. Without a check, a chained `--resume --amp` job resubmitted after this change
would silently continue an fp16 run in bf16. A helper used by `_maybe_resume`:

```python
def _checkpoint_amp(extra):
    """AMP mode a checkpoint was trained with; older ones only reveal fp16 via their scaler state."""
    return extra.get("amp", "fp16" if extra.get("scaler") else None)
```

If it differs from `self.amp`, print a clear warning that the run continues in a different
precision. An old fp16 scaler state is ignored under bf16.

## Step 4: tests (`tests/test_trainer.py`)

- Parser: no flag gives `None`, `--amp` gives `"bf16"`, `--amp fp16` gives `"fp16"`,
  `--amp fp32` is rejected, `--amp --compile` keeps `--compile`.
- `BaseTrainer(amp="fp32")` raises `ValueError`.
- CUDA only (`pytest.mark.skipif(not torch.cuda.is_available())`; they skip on a laptop and
  run on the cluster):
  - `amp="bf16"`: the scaler is disabled, and the existing `_poison_grad_trainer` scenario
    aborts with a report / skips with finite weights. This guards the safety-critical line.
  - `amp="fp16"`: the scaler is enabled and its state is saved and restored.
  - Resuming with a different precision warns, including from an old checkpoint that has
    `"scaler"` but no `"amp"`.

## Step 5: docs, configs and scripts

- `--amp` help text (Step 2).
- The `amp:` line of the 5 train templates: `(str, none)  mixed precision: true/bf16 = bf16
  autocast; fp16 = fp16 autocast + gradient scaler`.
- `README.md` (the `--compile`, `--amp`, `--channels_last` paragraph).
- The sbatch comments and `scripts/smoke/smoke_resnet_vae.sh` usage comment.
- `scripts/smoke/smoke_ms2020_vae.sh` and the `ms2020_vae` template (`amp: true`) switch to
  bf16 with no edit; this is intended.

## Step 6: verification

1. `pytest` locally (CPU tests) and on a GPU node (CUDA tests).
2. Dtype trace on a GPU node with `scripts/trace_autocast_dtypes.py` (already in the repo),
   for both models and both dtypes, e.g.

   ```bash
   python scripts/trace_autocast_dtypes.py --model variable_rate_lossy_vae --preset c32_l15 --patchsize 256 --trace_dtype bf16 --trace_out trace_vr_bf16
   ```

   Check the summary against "Where bf16 still rounds" below.
3. A 1-2 epoch bf16 smoke run of each model with the sbatch recipe: `nonfinite_grad_steps=0`
   every epoch, and loss / rate / MSE on the same track as the fp16 run's first epochs.

## Step 7: rollout

- Runs in progress: pin `--amp fp16` in both sbatch files until they finish, so the chained
  `--resume` jobs end in the precision they started in.
- New runs: `--amp` (bf16) for both models, with a new `--checkpoint_dir` (Step 3).

## Step 8 (optional, separate change): fp32 for the rate and likelihood math

Precision is lost where a conv or linear writes its bf16 output; a later `.float()` cannot
recover it. Run these layers outside autocast, on fp32 inputs (they have few output
channels, so the cost is small):

- `variable_rate_lossy_vae`: the `prior` / `posterior` head convs of each
  `LatentVariableBlock`, and the final patch-upsample that emits `x_hat`.
- `resnet_vae`: `gen_net`, `inf_net`, `td_inf_net` (latent locations and scales), the image
  head (in `0_255` mode `(conv + 0.5) * 255` stays bf16, so `x_hat` is spaced ~1 grey level
  apart above 128), the `randn_like` noise, and the `DeepFactorized` matmuls.

This changes numbers (also under fp16), so it gets its own commit and a before/after check.

## Where bf16 still rounds

From PyTorch's CUDA autocast lists (v2.13 source, `aten/src/ATen/autocast_mode.h`): convs
and linears/matmuls run in bf16; `exp`, `log`, `pow` (so `x ** 2`), `softplus`, `sum`,
`logsumexp`, `layer_norm` and the losses run in fp32; everything else runs in its inputs'
dtype (bf16 with fp32 promotes to fp32; Python scalars do not promote). Weights, gradients,
Adam and the EMA stay fp32.

| | `variable_rate_lossy_vae` | `resnet_vae` |
|---|---|---|
| Residual stream | fp32 (the fp32 layer-scale `gamma` promotes it) | bf16 (convs, GDN, residual adds) |
| Latent means / scales | bf16 at the head conv, then fp32 (`linear_sqrt`, `softplus`) | locations bf16; scales fp32 (`softplus`) |
| KL / log-densities | fp32 | fp32, except `q_loc − p_loc` in the non-AR levels (bf16) |
| Reparameterization noise | fp32 | bf16 |
| Top-level prior | none | `DeepFactorized` matmuls bf16, the rest fp32 |
| `x_hat` | bf16 | bf16 |
| Loss, rate, MSE | fp32 | fp32 |

The variable-rate model is mostly bf16-safe as is; `resnet_vae` has more exposure. Step 8
addresses both.

## Build order

1. Steps 1-3 (trainer, CLI, resume check), about 40 lines, mostly in `trainer.py`.
2. Step 4 (tests).
3. Step 5 (docs, templates, scripts).
4. Step 6 (verification on the cluster).
5. Step 7 (sbatch rollout).
6. Step 8 later, as its own change.
