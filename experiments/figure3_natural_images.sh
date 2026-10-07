#!/usr/bin/env bash
# Natural-image R-D upper bound figures from the TRAINED ResNet-VAE checkpoints
# (paper Sec. 6.4: Fig. 3 right = Kodak, Fig. 11 = Tecnick).
#
# 1. For every lambda found in $CKPT_DIR, evaluates the checkpoint on Kodak and
#    Tecnick with evaluation/eval_ub.py (full images, uint8 reconstructions),
#    unless that result .npz already exists in $RESULTS (FORCE_EVAL=1 redoes it).
# 2. Plots, one point per lambda, into $OUT_DIR (default results/):
#      qr_kodak.png      quality-rate: PSNR vs bpp (as in the paper's Fig. 3)
#      rd_kodak.png      rate-distortion: bpp vs MSE ([0, 255] scale)
#      qr_tecnick.png, rd_tecnick.png   the same on Tecnick (Fig. 11), if available
#
# Run:   bash experiments/figure3_natural_images.sh
# Env:   CKPT_DIR (default checkpoints/natural_images_resnet_vae_train)
#        RESULTS  (default results/img_compression)
#        KODAK    (default data/kodak)
#        TECNICK  (default data/Tecnick_TESTIMAGES/TESTIMAGES/RGB/RGB_OR_1200x1200)
#        OUT_DIR  (default results)
#        FORCE_EVAL=1 to re-evaluate even if results exist; DEVICE (e.g. cuda:1)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"  # works without `pip install -e .`

CKPT_DIR="${CKPT_DIR:-checkpoints/natural_images_resnet_vae_train}"
RESULTS="${RESULTS:-results/img_compression}"
KODAK="${KODAK:-data/kodak}"
TECNICK="${TECNICK:-data/Tecnick_TESTIMAGES/TESTIMAGES/RGB/RGB_OR_1200x1200}"
OUT_DIR="${OUT_DIR:-results}"
FORCE_EVAL="${FORCE_EVAL:-0}"
DEV=${DEVICE:+--device $DEVICE}

# Must match the training architecture (configs/images/resnet_vae_train_ub.yaml).
arch="--model resnet_vae --latent_channels 4,8,16,32,64,128 --ar_prior_levels 4 --ar_slices 8 --num_filters 256"
run_suffix="-F=256-C=4_8_16_32_64_128-arlv=4-arsl=8"

lambdas=$(ls -d "$CKPT_DIR"/rdub-model=resnet_vae-lambda=*"$run_suffix" 2>/dev/null \
  | sed -E 's/.*lambda=([^-]+)-F=.*/\1/' | sort -g)
[ -n "$lambdas" ] || { echo "No resnet_vae runs found in $CKPT_DIR"; exit 1; }
echo "lambdas: $(echo $lambdas)"

datasets=("kodak:$KODAK")
if [ -d "$TECNICK" ]; then datasets+=("tecnick:$TECNICK"); else echo "(no Tecnick at $TECNICK; skipping it)"; fi

for lamb in $lambdas; do
  for entry in "${datasets[@]}"; do
    name=${entry%%:*}; path=${entry#*:}
    out="$RESULTS/rdub-model=resnet_vae-lambda=$lamb-dataset=$name.npz"
    if [ -f "$out" ] && [ "$FORCE_EVAL" != "1" ]; then
      echo "[skip] $out exists"
      continue
    fi
    python evaluation/eval_ub.py $DEV $arch --lambda "$lamb" --checkpoint_dir "$CKPT_DIR" \
      --dataset "$path" --results_dir "$RESULTS"
  done
done

for entry in "${datasets[@]}"; do
  name=${entry%%:*}
  python evaluation/plot_qr.py --models resnet_vae --dataset "$name" --results_dir "$RESULTS" \
    --out "$OUT_DIR/smoke_qr_$name.png"
  python evaluation/plot_qr.py --models resnet_vae --dataset "$name" --results_dir "$RESULTS" \
    --out "$OUT_DIR/smoke_rd_$name.png" --rd
done
