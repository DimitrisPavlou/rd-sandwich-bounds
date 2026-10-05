#!/usr/bin/env bash
# Natural images (paper Sec. 6.4, Fig. 3/11): ResNet-VAE upper bound, trained on
# COCO patches and evaluated on Kodak.
set -e
cd "$(dirname "$0")/.."

echo "=== Natural images (COCO train -> Kodak/Tecnick) ==="
echo "First: download+prep COCO per README.md's 'Data preparation' section, then:"
echo "  wget -P /tmp http://images.cocodataset.org/zips/train2017.zip && unzip /tmp/train2017.zip -d /tmp"
echo "  python scripts/prepare_imgs.py '/tmp/train2017/*.jpg' data/my_coco_train2017/"
for lamb in 0.005 0.01 0.02 0.04 0.08 0.16; do
  python train/train_image_ub.py -V --model resnet_vae --command train \
    --dataset data/my_coco_train2017 --patchsize 256 \
    --latent_channels 4,8,16,32,64,128 --ar_prior_levels 4 --ar_slices 8 --num_filters 256 \
    --lambda $lamb --checkpoint_dir checkpoints/img_compression \
    --batchsize 8 --epochs 600 --warmup 400 --patience 20

  python train/train_image_ub.py -V --model resnet_vae --command eval \
    --dataset data/kodak \
    --latent_channels 4,8,16,32,64,128 --ar_prior_levels 4 --ar_slices 8 --num_filters 256 \
    --lambda $lamb --checkpoint_dir checkpoints/img_compression --results_dir results/img_compression
done
