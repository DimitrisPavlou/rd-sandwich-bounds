#!/usr/bin/env bash
# SMOKE TEST: the FULL n=1000 Gaussian R-D upper-bound sweep, cut to 2 epochs.
#
# Same source, models and settings as the real sweep (configs/gaussian_ub.yaml and
# configs/gaussian_ub_zy.yaml): the paper's fixed n=1000 Gaussian, the MLP beta-VAE with a
# decoder at latent_dim 400/600/800 plus the Z == Y (no decoder) model, 7 lambdas,
# batch 64, lr 5e-4, 1000 steps per epoch. Only --epochs differs (2 instead of 80).
# For every run: train_ub -> eval_ub (sampled (D, R) +/- 95% CI) -> then plot_rdub.
# Outputs go to checkpoints/smoke/gaussian_ub and results/smoke/gaussian_ub (wiped first).
#
# Run:   bash scripts/smoke/smoke_gaussian_ub.sh
#        DRY_RUN=1 bash scripts/smoke/smoke_gaussian_ub.sh       # only print the commands
# Env:   EPOCHS (2), LAMBDAS ("0.3 1 3 10 30 100 300"), LATENT_DIMS ("400 600 800"),
#        DEVICE (e.g. cuda:1)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"  # works without `pip install -e .`

EPOCHS="${EPOCHS:-2}"
LAMBDAS="${LAMBDAS:-0.3 1 3 10 30 100 300}"
LATENT_DIMS="${LATENT_DIMS:-400 600 800}"
DEV=${DEVICE:+--device $DEVICE}
n=1000
gp=data/gaussian/gaussian_params-dim=$n.npz
CKPT=checkpoints/smoke/gaussian_ub
RES=results/smoke/gaussian_ub

run() { echo "+ $*"; if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi; }
[ "${DRY_RUN:-0}" = 1 ] || rm -rf "$CKPT" "$RES"
[ -f "$gp" ] || run python scripts/gen_gaussian_params.py --save_dir data/gaussian --dim $n

data="--dataset gaussian --gparams_path $gp"
common="--model mlp_vae --prior_type gmm_1 --posterior_type gaussian --encoder_activation none --rpd"
optim="--epochs $EPOCHS --steps_per_epoch 1000 --lr 5.0e-4 --batchsize 64 --checkpoint_dir $CKPT"
evalargs="--checkpoint_dir $CKPT --results_dir $RES --batchsize 1024 --num_batches 20"

for lamb in $LAMBDAS; do
  echo "=== [gaussian UB] Z == Y (no decoder), lambda=$lamb ==="
  zy="$common --decoder_activation none --decoder_units 0"
  run python train/train_ub.py -V $DEV $data $zy --lambda "$lamb" $optim
  run python evaluation/eval_ub.py $DEV $data $zy --lambda "$lamb" $evalargs

  for ld in $LATENT_DIMS; do
    echo "=== [gaussian UB] decoder, latent_dim=$ld, lambda=$lamb ==="
    dec="$common --latent_dim $ld --decoder_activation leaky_relu --decoder_units $n"
    run python train/train_ub.py -V $DEV $data $dec --lambda "$lamb" $optim
    run python evaluation/eval_ub.py $DEV $data $dec --lambda "$lamb" $evalargs
  done
done

run python evaluation/plot_rdub.py --checkpoint_dir $CKPT --gparams_path $gp --out $RES/gaussian_ub.png
echo "=== gaussian UB smoke test PASSED (figure in $RES) ==="
