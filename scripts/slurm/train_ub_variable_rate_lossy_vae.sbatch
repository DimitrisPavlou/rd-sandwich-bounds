#!/bin/bash
# --- VUB-HPC (Hydra) GPU job -- variable-rate lossy VAE R-D UPPER BOUND ---------
# Duan et al. (ICIP 2023) variable-rate ResNet-VAE, preset c32_l15 (C=32, 15 latents,
# 18.6M params: about the ResNet-VAE's 15.7M). Shared recipe of
# configs/images/variable_rate_lossy_vae_train_ub.yaml (Duan's), except for the preset,
# --amp and --grad_clip. Same budget and schedule as train_ub_resnet_vae.sbatch.
#
# ONE run covers every lambda in [4, 2048] (sampled log-uniformly per image), so
# this is a single job, not an array.
#
# Submit:  sbatch scripts/slurm/train_ub_variable_rate_lossy_vae.sbatch
# Chain:   sbatch --dependency=afterany:<JOBID> scripts/slurm/train_ub_variable_rate_lossy_vae.sbatch
# Watch:   mysqueue
#
# 300 epochs = 1,108,800 steps; --checkpoint_interval + --resume let 48 h jobs chain.
# Requires raw COCO train2017 at the --dataset path below.

#SBATCH --job-name=rdub_vr_lossy_vae
#SBATCH --partition=hopper_gpu      # Nvidia H100 (test/debug: ampere_gpu A100, or -M anansi -p ada_gpu)
#SBATCH --gres=gpu:1                # 1 GPU (training is single-GPU, no DDP)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # >= num_workers (16)
#SBATCH --time=48:00:00             # resubmit to continue (--resume)
#SBATCH --output=logs/%x_%j.out     # %x=job-name %j=job-id
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# --- environment (edit the two paths to match your setup) --------------------
module load Miniconda3/23.9.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VSC_SCRATCH/.conda_envs/main"

cd "$VSC_SCRATCH/rd-sandwich-bounds"     # repo root, where train/ and data/ live
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"  # import rdsandwich without `pip install -e .`
mkdir -p logs

# --- sanity checks -----------------------------------------------------------
echo "=== host/job ==="
echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "=== Python ==="
which python && python --version
echo "=== PyTorch / CUDA ==="

# --- launch -------------------------------------------------------------------
args=(
  # ===== 1. data =============================================================
  --dataset /data/brussel/vo/000/bvo00053/vsc11860/coco_train2017    # raw COCO train2017 (118,287 images)
  --patchsize 256
  --hflip
  --small_image_mode reflect_pad      # mirror-pad images smaller than 256, as Duan et al.
  --num_workers 16                    # full COCO is decoded on the fly (too big to preload)

  # ===== 2. model ============================================================
  --model variable_rate_lossy_vae
  --preset c32_l15                    # C=32, latents 1,2,3,4,5 (18.6M params; not in the paper)
  --lmb_range 4.0,2048.0              # training lambda range (log-uniform)

  # ===== 3. training =========================================================
  --checkpoint_dir checkpoints/lvae_variable_rate_full_coco
  --batchsize 32
  --epochs 300                        # 3,696 steps/epoch -> 1,108,800 steps
  --lr 2.0e-4
  --lr_schedule const-cos             # constant for 150 epochs, then cosine to 0.01x
  --grad_clip 500.0                   # spike guard only: probe grad norms 50-280 over 800 steps (Duan: 2.0)
  --ema 0.9999                        # evaluated by default
  --ema_warmup 10000
  --checkpoint_interval 10            # resumable every 10 epochs
  --resume
  --amp                               # mixed precision (Duan: fp32); same for both models
  --verbose
  --compile
)

python train/train_ub.py "${args[@]}"
