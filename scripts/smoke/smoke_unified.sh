#!/usr/bin/env bash
# SMOKE TEST: the shared-recipe pipeline (plan.md) for BOTH models, in minutes.
#
# Same recipe flags as configs/images/resnet_vae_train_ub_pm1.yaml and
# configs/images/variable_rate_lossy_vae_train_ub.yaml (pm1 loss, const-cos LR,
# EMA, grad clip 2.0, h-flip, reflect padding, per-epoch checkpoints + resume), but
# tiny models, 128-px patches and a handful of images. For each model:
#   train_ub (EPOCHS epochs) -> train_ub --resume (one more epoch)
#   -> eval_ub on Kodak (EMA weights) at a few lambdas
# then plot_qr (Q-R and R-D) and plot_image_ub_training.
# Outputs go to checkpoints/smoke/unified and results/smoke/unified (wiped first).
#
# Run:   bash scripts/smoke/smoke_unified.sh
#        DRY_RUN=1 bash scripts/smoke/smoke_unified.sh          # only print the commands
# Env:   TRAIN_DATA (data/coco_train2017), KODAK (data/kodak), MAX_IMAGES (16),
#        EPOCHS (2), COMPILE (0; 1 adds --compile), DEVICE (e.g. cuda:1)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"  # works without `pip install -e .`

TRAIN_DATA="${TRAIN_DATA:-data/coco_train2017}"
KODAK="${KODAK:-data/kodak}"
MAX_IMAGES="${MAX_IMAGES:-16}"
EPOCHS="${EPOCHS:-2}"
DEV=${DEVICE:+--device $DEVICE}
CKPT=checkpoints/smoke/unified
RES=results/smoke/unified

run() { echo "+ $*"; if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi; }
[ "${DRY_RUN:-0}" = 1 ] || rm -rf "$CKPT" "$RES"

# --- 1. data (as the recipe, but few images and small patches) ---
data="--dataset $TRAIN_DATA --max_images $MAX_IMAGES --patchsize 128 --hflip --small_image_mode reflect_pad"
# --- 3. training (the recipe's optimizer settings; tiny batch) ---
train="--batchsize 4 --lr 2.0e-4 --lr_schedule const-cos --grad_clip 2.0 --ema 0.9999 --ema_warmup 10000
       --checkpoint_interval 1 --checkpoint_dir $CKPT -V"
[ "${COMPILE:-0}" = 1 ] && train+=" --compile"

# --- 2. model: tiny versions of both models ---
resnet="--model resnet_vae --image_range pm1 --latent_channels 4,8,16 --ar_prior_levels 1 --ar_slices 2 --num_filters 16"
duan="--model variable_rate_lossy_vae --preset tiny"

# ResNet-VAE: one run per lambda (pm1 units)
for lamb in 64 512; do
  echo "=== [resnet_vae pm1] lambda=$lamb: train $EPOCHS epochs, then resume for 1 more ==="
  run python train/train_ub.py $DEV $data $resnet --lambda "$lamb" $train --epochs "$EPOCHS"
  run python train/train_ub.py $DEV $data $resnet --lambda "$lamb" $train --epochs $((EPOCHS + 1)) --resume
  run python evaluation/eval_ub.py $DEV $resnet --lambda "$lamb" --dataset "$KODAK" --max_images 2 \
    --checkpoint_dir $CKPT --results_dir $RES
done

# Variable-rate model: one run, evaluated at several lambdas
echo "=== [variable_rate_lossy_vae tiny] train $EPOCHS epochs, then resume for 1 more ==="
run python train/train_ub.py $DEV $data $duan $train --epochs "$EPOCHS"
run python train/train_ub.py $DEV $data $duan $train --epochs $((EPOCHS + 1)) --resume
for lamb in 8 64 512; do
  run python evaluation/eval_ub.py $DEV $duan --lambda "$lamb" --dataset "$KODAK" --max_images 2 \
    --checkpoint_dir $CKPT --results_dir $RES
done

ds=$(python -c "from rdsandwich.data.datasets import dataset_name; print(dataset_name('$KODAK'))")
run python evaluation/plot_qr.py --dataset "$ds" --results_dir $RES --out $RES/qr_$ds.png
run python evaluation/plot_qr.py --dataset "$ds" --results_dir $RES --out $RES/rd_$ds.png --rd
run python evaluation/plot_image_ub_training.py --checkpoint_dir $CKPT --last_n 1 --out_prefix $RES/train
echo "=== unified smoke test PASSED (figures in $RES) ==="
