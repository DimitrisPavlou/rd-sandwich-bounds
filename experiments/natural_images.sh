#!/usr/bin/env bash
# Natural images (paper Sec. 6.4, Fig. 3/11): ResNet-VAE upper bound, trained on
# COCO patches and evaluated on Kodak.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"  # works without `pip install -e .`

echo "=== Natural images (COCO train -> Kodak/Tecnick) ==="
echo "First: download+prep COCO per README.md's 'Data preparation' section, then:"
echo "  wget -P /tmp http://images.cocodataset.org/zips/train2017.zip && unzip /tmp/train2017.zip -d /tmp"
echo "  python scripts/prepare_imgs.py '/tmp/train2017/*.jpg' data/my_coco_train2017/"
arch="--model resnet_vae --latent_channels 4,8,16,32,64,128 --ar_prior_levels 4 --ar_slices 8 --num_filters 256"
for lamb in 0.005 0.01 0.02 0.04 0.08 0.16; do
  python train/train_ub.py -V $arch --lambda $lamb --checkpoint_dir checkpoints/img_compression \
    --dataset data/my_coco_train2017 --patchsize 256 --num_workers 4 \
    --batchsize 8 --lr 1e-4 --epochs 600 --lr_schedule plateau --warmup 400 --patience 20 \
    --checkpoint_interval 10

  python evaluation/eval_ub.py -V $arch --lambda $lamb --checkpoint_dir checkpoints/img_compression \
    --dataset data/kodak --results_dir results/img_compression
done
