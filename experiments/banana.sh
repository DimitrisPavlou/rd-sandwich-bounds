#!/usr/bin/env bash
# Reproduces the banana-source sandwich bounds and intrinsic-vs-nominal-dimension
# study (paper Sec. 6.3 / Figs. 7, 8).
set -e
cd "$(dirname "$0")/.."

python - <<'PY'
import numpy as np
from rdsandwich.dataloader import BananaSource
np.save("data/banana-dim=2-samples=100000.npy", BananaSource().sample(100000).numpy())
PY

python scripts/run_ba.py --samples data/banana-dim=2-samples=100000.npy \
  --save_dir checkpoints/banana/BA --bins 80 --steps 2000 --tol 1e-5 --lamb 1.0

declare -A nn_sizes_ub=( [4]=400 [16]=1600 [100]=2000 [500]=2000 )
for n in 4 16 100 500; do
  nn_size=${nn_sizes_ub[$n]}
  for lamb in 0.1 0.3 1 3 10 30 100 300 1000; do
    python train/train_rdub.py -V --checkpoint_dir checkpoints/banana --dataset banana --data_dim $n \
      --latent_dim $n --prior_type maf --maf_stacks 3 --posterior_type gaussian \
      --encoder_units $nn_size,$nn_size --decoder_units $nn_size,$nn_size \
      --encoder_activation softplus --decoder_activation softplus --nats \
      --lambda $lamb --epochs 100 --steps_per_epoch 1000 --lr 5e-4 --batchsize 1024
  done
done

declare -A nn_sizes_lb=( [2]=200 [4]=400 [16]=1000 [100]=1000 [500]=1000 )
for n in 2 4 16 100 500; do
  nn_size=${nn_sizes_lb[$n]}
  for lamb in 0.1 0.3 1 3 10 30 100 300; do
    python train/train_rdlb.py -V --dataset banana --data_dim $n --checkpoint_dir checkpoints/banana \
      --model mlp --units $nn_size,$nn_size,$nn_size --lamb $lamb \
      --command train --batchsize 1024 --num_Ck_samples 2 --last_step 4000 --checkpoint_interval 4000 \
      --y_init quick --y_quick_topn 20 --lr 1e-3
  done
done
