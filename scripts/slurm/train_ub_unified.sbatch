#!/bin/bash
# --- VUB-HPC (Hydra) GPU job -- shared-recipe R-D UPPER BOUND training ----------
# Same environment as train_ub_resnet_vae.sbatch, but every setting comes from a
# YAML config: array task i runs the i-th run of that config
# (scripts/run_sweep.py --index i), so the YAML is the only place the recipe lives.
#
# Submit (one lambda per array task for the fixed-rate model, one task for Duan's):
#   sbatch --array=0-5%2 --export=ALL,CONFIG=configs/images/resnet_vae_train_ub_pm1.yaml \
#       scripts/slurm/train_ub_unified.sbatch
#   sbatch --array=0 --export=ALL,CONFIG=configs/images/variable_rate_lossy_vae_train_ub.yaml \
#       scripts/slurm/train_ub_unified.sbatch
# Preview which run an index maps to:
#   python scripts/run_sweep.py --config <CONFIG> --dry-run
#
# The configs set resume: true and checkpoint_interval: 1, so resubmitting the same
# array task after the 48 h wall continues from the last finished epoch.
#
# Requires data/coco_train2017 (raw COCO train2017) under the repo directory.

#SBATCH --job-name=rdub_unified
#SBATCH --partition=hopper_gpu      # Nvidia H100 (test/debug: ampere_gpu A100, or -M anansi -p ada_gpu)
#SBATCH --gres=gpu:1                # 1 GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # >= num_workers in the config (16)
#SBATCH --time=48:00:00             # resubmit to continue (resume: true)
#SBATCH --output=logs/%x_%A_%a.out  # %x=job-name %A=array-job-id %a=array-task-id
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
: "${CONFIG:?set CONFIG, e.g. sbatch --export=ALL,CONFIG=configs/<name>.yaml ...}"

# --- environment (edit the two paths to match your setup) --------------------
module load Miniconda3/23.9.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VSC_SCRATCH/.conda_envs/main"

cd "$VSC_SCRATCH/rd-sandwich-bounds"     # repo root, where train/ and data/ live
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"  # import rdsandwich without `pip install -e .`
mkdir -p logs

# --- sanity checks -----------------------------------------------------------
echo "=== host/task ==="
echo "host=$(hostname)  config=$CONFIG  task=$SLURM_ARRAY_TASK_ID"
echo "=== PyTorch / CUDA ==="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# --- launch: the SLURM_ARRAY_TASK_ID-th run of the config --------------------
python scripts/run_sweep.py --config "$CONFIG" --index "$SLURM_ARRAY_TASK_ID"
