#!/usr/bin/env bash
# Reproduces the particle-physics / speech sandwich-bound sweeps (paper Sec. 6.2 / Fig. 2b,c).
#
# Prerequisites (see the original README.md's "Data preparation" section):
#   - data/physics/ppzee-split={train,test}.npy   (from Howard et al., 2021)
#   - data/speech/*.npy                            (from the Free Spoken Digit Dataset,
#                                                     via data/speech/create_data.py)
set -e
cd "$(dirname "$0")/.."

for dataset_name in physics speech; do
  if [ "$dataset_name" = "physics" ]; then
    train_path=data/physics/ppzee-split=train.npy
    n=16
  else
    train_path=data/speech/train.npy
    n=33
  fi

  for lamb in 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10; do
    python train/train_rdub.py -V --dataset $train_path --data_dim $n --latent_dim $n \
      --checkpoint_dir checkpoints/$dataset_name --prior_type maf --maf_stacks 3 \
      --posterior_type gaussian --encoder_units 500,500 --decoder_units 500,500 \
      --encoder_activation softplus --decoder_activation softplus --nats \
      --lambda $lamb --epochs 100 --steps_per_epoch 1000 --lr 5e-4 --batchsize 256
  done

  for lamb in 100 300 1000 3000; do
    python train/train_rdlb.py -V --dataset $train_path --data_dim $n \
      --checkpoint_dir checkpoints/$dataset_name --model mlp --units 200,200,200 --lamb $lamb \
      --command train --num_Ck_samples 1 --batchsize 2048 --last_step 4000 --checkpoint_interval 4000 \
      --y_init quick --y_quick_topn 10 --lr 5e-4
  done
done

echo "For a ground-truth 2D-marginal comparison via Blahut-Arimoto, extract the two least-"
echo "correlated coordinates (see the paper's Appendix A.5.3/A.5.4) and run e.g.:"
echo "  python scripts/run_ba.py --samples data/physics/2d_marginal.npy --lamb 1.0 --bins 80 --steps 2000"
