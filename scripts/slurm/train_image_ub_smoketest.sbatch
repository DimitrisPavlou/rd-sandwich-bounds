#!/bin/bash
# --- SMOKE TEST: does the whole pipeline run on a GPU node? -------------------
# A short job (1 GPU, FULL dataset, 3 epochs, real batchsize 16) that exercises
# the SAME path as the real run -- env, GPU visibility, data loading, model,
# torch.compile, AMP, checkpoint saving -- AND measures the true per-epoch time.
# Read the "time=...s (...s/step)" from epoch 1+ (epoch 0 includes compile warmup)
# to extrapolate the full run:  hours/lambda ~= 527 * 2000 * (s/step) / 3600.
#
# Submit:  sbatch scripts/slurm/train_image_ub_smoketest.sbatch
# Watch:   mysqueue   (then read logs/rdub_smoketest_*.out)

#SBATCH --job-name=rdub_smoketest
#SBATCH --partition=ampere_gpu      # A100 (easier to queue). H100: hopper_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00             # generous; the run itself is minutes
#SBATCH --output=logs/%x_%j.out     # %j = job id
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# --- environment -------------------------------------------------------------
module load Miniconda3/23.9.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VSC_SCRATCH/.conda_envs/main"

cd "$VSC_SCRATCH/rd-sandwich-bounds"
mkdir -p logs

# --- sanity checks -----------------------------------------------------------
echo "=== host/task ==="
echo "host=$(hostname)"
echo "=== Python / PyTorch / CUDA ==="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "=== data ==="
ls data/my_coco_train2017 | wc -l

# --- tiny training run (same flags as the real job, but scaled down) ---------
python train/train_image_ub.py \
  --model resnet_vae \
  --command train \
  --dataset data/my_coco_train2017 \
  --checkpoint_dir checkpoints/smoketest \
  --latent_channels 4,8,16,32,64,128 \
  --ar_prior_levels 4 \
  --ar_slices 8 \
  --num_filters 256 \
  --patchsize 256 \
  --batchsize 16 \
  --num_workers 8 \
  --epochs 3 \
  --warmup 1 \
  --patience 60 \
  --checkpoint_interval 1 \
  --lr 1.0e-4 \
  --lambda 0.01 \
  --verbose \
  --amp \
  --compile \
  --preload

echo "=== smoke test finished; checkpoints written: ==="
ls -la checkpoints/smoketest/*/ 2>/dev/null || echo "(no checkpoint found -- investigate)"
