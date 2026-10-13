#!/usr/bin/env bash
# Natural-image R-D upper bound figures from the TRAINED ResNet-VAE checkpoints
# (paper Sec. 6.4: Fig. 3 right = Kodak, Fig. 11 = Tecnick).
#
# 1. Evaluates every resnet_vae run found in $CKPT_DIR on Kodak and Tecnick with
#    evaluation/evaluate.py (full images), unless its result file already exists in
#    $RESULTS (FORCE_EVAL=1 redoes it).
# 2. Plots, one point per lambda and PSNR / MSE of 8-bit reconstructions, into $OUT_DIR
#    (default results/) with evaluation/plot.py:
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

# The architecture of configs/images/resnet_vae_train_ub.yaml (the 0_255 runs)
run_suffix="-F=256-C=4_8_16_32_64_128-arlv=4-arsl=8"
ls -d "$CKPT_DIR"/rdub-model=resnet_vae-lambda=*"$run_suffix" >/dev/null 2>&1 \
  || { echo "No resnet_vae runs found in $CKPT_DIR"; exit 1; }

datasets="kodak: \"$KODAK\""; names="kodak"
if [ -d "$TECNICK" ]; then datasets+=", tecnick: \"$TECNICK\""; names+=", tecnick"
else echo "(no Tecnick at $TECNICK; skipping it)"; fi

mkdir -p "$RESULTS"
cat > "$RESULTS/eval_resnet_vae.yaml" <<EOF
name: resnet_vae
label: "proposed \$R_U(D)\$ (ResNet-VAE)"
checkpoints: ["$CKPT_DIR/rdub-model=resnet_vae-lambda=*$run_suffix"]
datasets: {$datasets}
results_dir: "$RESULTS"
device: ${DEVICE:-null}
EOF
cat > "$RESULTS/plot_figure3.yaml" <<EOF
out_dir: "$OUT_DIR"
curves: [resnet_vae]
datasets: [$names]
results_dir: "$RESULTS"
distortion: uint8
combined: false
EOF
force=(); if [ "$FORCE_EVAL" = 1 ]; then force=(--force); fi
python evaluation/evaluate.py --config "$RESULTS/eval_resnet_vae.yaml" "${force[@]}"
python evaluation/plot.py --config "$RESULTS/plot_figure3.yaml"
