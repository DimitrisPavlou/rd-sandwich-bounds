#!/usr/bin/env bash
# GAN-generated images (paper Sec. 6.3, Fig. 3) and natural images (Sec. 6.4, Fig. 3/11).
#
# Requires: pip install -e ".[images]"   (compressai, pytorch-pretrained-biggan)
set -e
cd "$(dirname "$0")/.."

echo "=== GAN-generated (basenji) images ==="
for d in 2 4; do
  for lamb in 5e-4 3e-4 1e-4 3e-5 1e-5 3e-6 1e-6; do
    python train/train_resnet_vae.py -V --dataset basenji --data_dim $d --img_dim 128 --flat_z0 \
      --latent_channels 4,8,16,32,64,128 --num_filters 256 --lambda $lamb \
      --checkpoint_dir checkpoints/gan/dataset=basenji-data_dim=$d \
      --batchsize 8 --epochs 100 --steps_per_epoch 200
  done

  python train/train_rdlb.py -V --dataset basenji --data_dim $d \
    --checkpoint_dir checkpoints/gan/dataset=basenji-data_dim=$d \
    --model cnn --units 4,8,16,1024 --kernel_dims 9,5,3 --lamb 4000 \
    --command train --batchsize 512 --num_Ck_samples 2 --last_step 4000 --checkpoint_interval 1000 \
    --y_init quick --y_quick_topn 3 --lr 1e-4
done

echo "=== Natural images (COCO train -> Kodak/Tecnick) ==="
echo "First: download+prep COCO per README.md's 'Data preparation' section, then:"
echo "  wget -P /tmp http://images.cocodataset.org/zips/train2017.zip && unzip /tmp/train2017.zip -d /tmp"
echo "  python scripts/prepare_imgs.py '/tmp/train2017/*.jpg' data/my_coco_train2017/"
for lamb in 0.005 0.01 0.02 0.04 0.08 0.16; do
  python train/train_resnet_vae.py -V --dataset data/my_coco_train2017 --img_dim 256 \
    --latent_channels 4,8,16,32,64,128 --num_filters 256 --lambda $lamb \
    --checkpoint_dir checkpoints/img_compression --batchsize 8 --epochs 100 --steps_per_epoch 1000
done
