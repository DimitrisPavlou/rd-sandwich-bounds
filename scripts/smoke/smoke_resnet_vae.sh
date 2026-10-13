#!/usr/bin/env bash
# SMOKE TEST: the FULL natural-image ResNet-VAE pipeline, cut to 2 epochs.
#
# Same model, data and optimization settings as the real run
# (scripts/slurm/train_ub_resnet_vae.sbatch / configs/images/resnet_vae_train_ub.yaml):
# full-size ResNet-VAE, the whole COCO training set, 256x256 patches, batch 16,
# all 6 lambdas, torch.compile, preloaded images. Only --epochs differs (2 instead of 2000).
# For every lambda: train_ub; then evaluation/evaluate.py on Kodak (and Tecnick, if present) and evaluation/plot.py.
# Outputs go to checkpoints/smoke/resnet_vae and results/smoke/resnet_vae (wiped first),
# so the real checkpoints are never touched.
#
# Run:   bash scripts/smoke/smoke_resnet_vae.sh
#        bash scripts/smoke/smoke_resnet_vae.sh --amp            # extra flags go to train_ub (--amp = bf16, --amp fp16)
#        DRY_RUN=1 bash scripts/smoke/smoke_resnet_vae.sh         # only print the commands
# Env:   EPOCHS (2), LAMBDAS ("0.005 0.01 0.02 0.04 0.08 0.16"), COMPILE (1), PRELOAD (1),
#        TRAIN_DATA (data/my_coco_train2017), KODAK (data/kodak),
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
CKPT=checkpoints/smoke/resnet_vae
RES=results/smoke/resnet_vae

run() { echo "+ $*"; if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi; }
[ "${DRY_RUN:-0}" = 1 ] || rm -rf "$CKPT" "$RES"

arch="--model resnet_vae --latent_channels 4,8,16,32,64,128 --ar_prior_levels 4 --ar_slices 8 --num_filters 256"
train="--dataset $TRAIN_DATA --patchsize 256 --batchsize 16 --num_workers 10 --lr 1.0e-4 --grad_clip 5000
       --lr_schedule plateau --warmup 1200 --patience 60 --checkpoint_interval 1 --checkpoint_dir $CKPT"
[ "${PRELOAD:-1}" = 1 ] && train+=" --preload"

datasets="kodak: \"$KODAK\""; names="kodak"
if [ -d "$TECNICK" ]; then datasets+=", tecnick: \"$TECNICK\""; names+=", tecnick"
else echo "(no Tecnick at $TECNICK; evaluating on Kodak only)"; fi

for lamb in $LAMBDAS; do
  echo "=== [resnet_vae] train $EPOCHS epochs, lambda=$lamb ==="
  run python train/train_ub.py -V $DEV $arch $train --lambda "$lamb" --epochs "$EPOCHS" "$@"
done

mkdir -p $RES
cat > $RES/eval.yaml <<EOF
name: resnet_vae
checkpoints: ["$CKPT/rdub-model=resnet_vae-lambda=*"]
datasets: {$datasets}
results_dir: $RES
device: ${DEVICE:-null}
EOF
cat > $RES/plot.yaml <<EOF
out_dir: $RES
curves: [resnet_vae]
datasets: [$names]
results_dir: $RES
EOF
run python evaluation/evaluate.py --config $RES/eval.yaml
run python evaluation/plot.py --config $RES/plot.yaml
echo "=== resnet_vae smoke test PASSED (figures in $RES) ==="
