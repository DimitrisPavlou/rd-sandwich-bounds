#!/usr/bin/env bash
# SMOKE TEST: the FULL natural-image Minnen & Singh 2020 beta-VAE pipeline, cut to 2 epochs.
#
# Same model, data and optimization settings as the real run
# (configs/natural_images_ms2020_vae_train.yaml): full-size MS2020-VAE (latent_depth 320,
# hyperprior_depth 192, 192 filters, 10 slices), the whole COCO training set, 256x256
# patches, batch 16, all 6 lambdas, torch.compile + AMP. Only --epochs differs (2 instead
# of 2000). For every lambda: train_ub -> eval_ub on Kodak (and Tecnick, if present)
# -> then plot_qr.
# Outputs go to checkpoints/smoke/ms2020_vae and results/smoke/ms2020_vae (wiped first),
# so the real checkpoints are never touched.
#
# Run:   bash scripts/smoke/smoke_ms2020_vae.sh
#        bash scripts/smoke/smoke_ms2020_vae.sh --grad_clip 5000  # extra flags go to train_ub
#        DRY_RUN=1 bash scripts/smoke/smoke_ms2020_vae.sh          # only print the commands
# Env:   EPOCHS (2), LAMBDAS ("0.005 0.01 0.02 0.04 0.08 0.16"), COMPILE (1), AMP (1),
#        PRELOAD (0), TRAIN_DATA (data/my_coco_train2017), KODAK (data/kodak),
#        TECNICK (data/Tecnick_TESTIMAGES/TESTIMAGES/RGB/RGB_OR_1200x1200), DEVICE (e.g. cuda:1)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"  # works without `pip install -e .`

EPOCHS="${EPOCHS:-2}"
LAMBDAS="${LAMBDAS:-0.005 0.01 0.02 0.04 0.08 0.16}"
TRAIN_DATA="${TRAIN_DATA:-data/my_coco_train2017}"
KODAK="${KODAK:-data/kodak}"
TECNICK="${TECNICK:-data/Tecnick_TESTIMAGES/TESTIMAGES/RGB/RGB_OR_1200x1200}"
DEV=${DEVICE:+--device $DEVICE}
CKPT=checkpoints/smoke/ms2020_vae
RES=results/smoke/ms2020_vae

run() { echo "+ $*"; if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi; }
[ "${DRY_RUN:-0}" = 1 ] || rm -rf "$CKPT" "$RES"

arch="--model ms2020_vae --latent_depth 320 --hyperprior_depth 192 --num_filters 192 --num_slices 10 --max_support_slices 5"
train="--dataset $TRAIN_DATA --patchsize 256 --batchsize 16 --num_workers 10 --lr 1.0e-4
       --lr_schedule plateau --warmup 1200 --patience 60 --checkpoint_interval 1 --checkpoint_dir $CKPT"
[ "${COMPILE:-1}" = 1 ] && train+=" --compile"
[ "${AMP:-1}" = 1 ] && train+=" --amp"
[ "${PRELOAD:-0}" = 1 ] && train+=" --preload"

evalsets=("$KODAK")
if [ -d "$TECNICK" ]; then evalsets+=("$TECNICK"); else echo "(no Tecnick at $TECNICK; evaluating on Kodak only)"; fi

for lamb in $LAMBDAS; do
  echo "=== [ms2020_vae] train $EPOCHS epochs, lambda=$lamb ==="
  run python train/train_ub.py -V $DEV $arch $train --lambda "$lamb" --epochs "$EPOCHS" "$@"
  for data in "${evalsets[@]}"; do
    echo "=== [ms2020_vae] eval on $data, lambda=$lamb ==="
    run python evaluation/eval_ub.py $DEV $arch --lambda "$lamb" --checkpoint_dir $CKPT \
      --dataset "$data" --results_dir $RES
  done
done

for data in "${evalsets[@]}"; do
  ds=$(python -c "from rdsandwich.data import dataset_name; print(dataset_name('$data'))")
  run python evaluation/plot_qr.py --models ms2020_vae --dataset "$ds" --results_dir $RES --out $RES/qr_$ds.png
  run python evaluation/plot_qr.py --models ms2020_vae --dataset "$ds" --results_dir $RES --out $RES/rd_$ds.png --rd
done
echo "=== ms2020_vae smoke test PASSED (figures in $RES) ==="
