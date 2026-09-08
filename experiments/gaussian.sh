#!/usr/bin/env bash
# Reproduces the n=1000 Gaussian sandwich-bound sweep (paper Sec. 6.1 / Fig. 2a).
set -e
cd "$(dirname "$0")/.."

n=1000
python scripts/gen_gaussian_params.py --save_dir data --dim $n

# --- Upper bound: with a decoder, varying latent_dim ---
for latent_dim in 400 500 600 800; do
  for lamb in 0.3 1 3 10 30 100 300; do
    python train/train_rdub.py -V --dataset gaussian --gparams_path data/gaussian_params-dim=$n.npz \
      --data_dim $n --latent_dim $latent_dim --checkpoint_dir checkpoints/gaussian \
      --prior_type gmm_1 --posterior_type gaussian \
      --encoder_activation none --decoder_activation leaky_relu --decoder_units $n \
      --lambda $lamb --rpd --epochs 80 --steps_per_epoch 1000 --lr 5e-4 --batchsize 64
  done
done

# --- Upper bound: Z == Y (no decoder) ---
for lamb in 0.3 1 3 10 30 100 300; do
  python train/train_rdub.py -V --dataset gaussian --gparams_path data/gaussian_params-dim=$n.npz \
    --data_dim $n --latent_dim $n --checkpoint_dir checkpoints/gaussian \
    --prior_type gmm_1 --posterior_type gaussian \
    --encoder_activation none --decoder_activation none --decoder_units 0 \
    --lambda $lamb --rpd --epochs 80 --steps_per_epoch 1000 --lr 5e-4 --batchsize 64
done

# --- Lower bound ---
nn_size_to_n_ratio=100
nn_size=$(( n * nn_size_to_n_ratio ))
for lamb in $(python -c "print(*[float(n)*x for x in [0.1,0.3,1,3,10]])"); do
  python train/train_rdlb.py -V --dataset gaussian --data_dim $n --checkpoint_dir checkpoints/gaussian \
    --model mlp --units $nn_size,$nn_size,$nn_size --lamb $lamb \
    --command train --num_Ck_samples 2 --batchsize 1024 --last_step 3000 --checkpoint_interval 3000 \
    --y_init quick --y_quick_topn 10 --lr 5e-4
done
