#!/bin/bash
# --- VUB-HPC (Hydra) GPU job — natural-images ResNet-VAE R-D UPPER BOUND ------
# Follows the VUB group template (Miniconda3 module + conda env in $VSC_SCRATCH,
# hopper_gpu / H100). Mirrors configs/natural_images_resnet_vae_train.yaml.
#
# Each array task runs ONE lambda on ONE GPU (training is single-GPU, no DDP),
# so all 6 lambdas sweep in parallel across the cluster.
#
# Submit:  sbatch scripts/slurm/train_image_ub.sbatch
# Watch:   mysqueue
#
# Requires COCO prepared under $VSC_SCRATCH/<code-dir>/data/my_coco_train2017.

#SBATCH --job-name=rdub_resnetvae
#SBATCH --partition=hopper_gpu      # Nvidia H100 (test/debug: ampere_gpu A100, or -M anansi -p ada_gpu)
#SBATCH --array=0-5                 # one task per lambda (6 values)
#SBATCH --gres=gpu:1                # 1 GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # >= num_workers (10); memory scales with cores
#SBATCH --time=48:00:00             # long full-budget run; --resume picks up if it hits the wall
#SBATCH --output=logs/%x_%A_%a.out  # %x=job-name %A=array-job-id %a=array-task-id
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# --- environment (edit the two paths to match your setup) --------------------
module load Miniconda3/23.9.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VSC_SCRATCH/.conda_envs/main"

cd "$VSC_SCRATCH/rd-sandwich-bounds"     # repo root, where train/ and data/ live
mkdir -p logs

# --- lambda sweep (matches the YAML 'sweep' block) ---------------------------
LAMBDAS=(0.005 0.01 0.02 0.04 0.08 0.16)
LAMBDA="${LAMBDAS[$SLURM_ARRAY_TASK_ID]}"

# --- sanity checks -----------------------------------------------------------
echo "=== host/task ==="
echo "host=$(hostname)  task=$SLURM_ARRAY_TASK_ID  lambda=$LAMBDA"
echo "=== Python ==="
which python && python --version
echo "=== PyTorch / CUDA ==="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# --- launch (fixed args mirror the YAML 'fixed' block) -----------------------
python train/train_image_ub.py \
  --model resnet_vae \
  --command train \
  --dataset data/my_coco_train2017 \
  --checkpoint_dir checkpoints/natural_images_resnet_vae_train \
  --latent_channels 4,8,16,32,64,128 \
  --ar_prior_levels 4 \
  --ar_slices 8 \
  --num_filters 256 \
  --patchsize 256 \
  --batchsize 16 \
  --num_workers 10 \
  --epochs 2000 \
  --warmup 1200 \
  --patience 60 \
  --checkpoint_interval 25 \
  --lr 1.0e-4 \
  --lambda "$LAMBDA" \
  --verbose \
  --amp \
  --compile \
  --preload \
  --resume
