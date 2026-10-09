#!/bin/bash
# --- VUB-HPC (Hydra) GPU job — natural-images ResNet-VAE R-D UPPER BOUND ------
# Yang & Mandt's ResNet-VAE (num_filters 256, 15.7M params) on raw COCO, trained with
# the SAME recipe, budget and schedule as train_ub_variable_rate_lossy_vae.sbatch, so
# the two R-D curves differ only because of the model. Shared recipe of
# configs/images/resnet_vae_train_ub_pm1.yaml, except for --amp and --grad_clip.
#
# Each array task runs ONE lambda on ONE GPU (training is single-GPU, no DDP).
# Lambdas are Duan's (pm1 units: KL nats/dim + lambda * MSE on [-1, 1]), so they
# are on the same scale as the variable-rate model's [4, 2048].
#
# Submit:  sbatch scripts/slurm/train_ub_resnet_vae.sbatch
# Chain:   sbatch --dependency=afterany:<ARRAY_JOBID> scripts/slurm/train_ub_resnet_vae.sbatch
# Watch:   mysqueue
#
# 300 epochs = 1,108,800 steps per lambda; --checkpoint_interval + --resume let 48 h jobs chain.
# Requires raw COCO train2017 at the --dataset path below.

#SBATCH --job-name=rdub_resnetvae
#SBATCH --partition=hopper_gpu      # Nvidia H100 (test/debug: ampere_gpu A100, or -M anansi -p ada_gpu)
#SBATCH --array=0-5%2               # all 6 lambdas, at most 2 running at once
#SBATCH --gres=gpu:1                # 1 GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # >= num_workers (16)
#SBATCH --time=48:00:00             # resubmit to continue (--resume)
#SBATCH --output=logs/%x_%A_%a.out  # %x=job-name %A=array-job-id %a=array-task-id
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# --- environment (edit the two paths to match your setup) --------------------
module load Miniconda3/23.9.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VSC_SCRATCH/.conda_envs/main"

cd "$VSC_SCRATCH/rd-sandwich-bounds"     # repo root, where train/ and data/ live
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"  # import rdsandwich without `pip install -e .`
mkdir -p logs

# --- lambda sweep (pm1 units, matches resnet_vae_train_ub_pm1.yaml) -----------
LAMBDAS=(8 22.6 64 181 512 1448)    # must cover every index in --array above
LAMBDA="${LAMBDAS[$SLURM_ARRAY_TASK_ID]}"

# --- sanity checks -----------------------------------------------------------
echo "=== host/task ==="
echo "host=$(hostname)  task=$SLURM_ARRAY_TASK_ID  lambda=$LAMBDA"
echo "=== Python ==="
which python && python --version
echo "=== PyTorch / CUDA ==="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# --- launch -------------------------------------------------------------------
args=(
  # ===== 1. data =============================================================
  --dataset /data/brussel/vo/000/bvo00053/vsc11860/coco_train2017    # raw COCO train2017 (118,287 images)
  --patchsize 256
  --hflip
  --small_image_mode reflect_pad      # mirror-pad images smaller than 256, as Duan et al.
  --num_workers 16                    # full COCO is decoded on the fly (too big to preload)

  # ===== 2. model ============================================================
  --model resnet_vae
  --image_range pm1                   # Duan's loss form and lambda units
  --latent_channels 4,8,16,32,64,128
  --ar_prior_levels 4
  --ar_slices 8
  --num_filters 256
  --lambda "$LAMBDA"

  # ===== 3. training =========================================================
  --checkpoint_dir checkpoints/resnet_vae_full_coco
  --batchsize 32
  --epochs 300                        # 3,696 steps/epoch -> 1,108,800 steps
  --lr 2.0e-4
  --lr_schedule const-cos             # constant for 150 epochs, then cosine to 0.01x
  --grad_clip 500.0                   # spike guard, as the lossy VAE; check clip_frac in epoch 0
  --ema 0.9999                        # evaluated by default
  --ema_warmup 10000
  --checkpoint_interval 1             # resumable every epoch
  --resume
  --amp fp16                          # mixed precision; same for both models.
                                      # fp16 so the running --resume chain ends as it started;
                                      # new runs: --amp (bf16) with a new --checkpoint_dir
  --compile                           # speed only
  --verbose
)

python train/train_ub.py "${args[@]}"
