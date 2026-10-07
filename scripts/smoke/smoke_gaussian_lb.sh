#!/usr/bin/env bash
# SMOKE TEST: the FULL Gaussian R-D lower-bound sweep, with a short training budget.
#
# Same sources, model and settings as the real sweep (configs/gaussian/mlp_train_eval_lb.yaml):
# standard Gaussians of dimension 2/4/8/16, 7 lambdas, a log-u MLP auto-sized to
# 2 x 20n SeLU units (paper A.5.2), k=1024, M=2 with the quick inner optimizer for
# training, then the exhaustive optimizer with M=5 for the reported estimate.
# The lower bound trains in steps, not epochs: only --last_step differs (200 instead of 3000).
# For every run: train_lb -> eval_lb (writes rd-*.npz) -> then plot_rdlb.
# Outputs go to checkpoints/smoke/gaussian_lb and results/smoke/gaussian_lb (wiped first).
#
# Run:   bash scripts/smoke/smoke_gaussian_lb.sh
#        DRY_RUN=1 bash scripts/smoke/smoke_gaussian_lb.sh       # only print the commands
# Env:   LAST_STEP (200), DIMS ("2 4 8 16"), LAMBDAS ("1 3 10 30 100 300 1000"),
#        DEVICE (e.g. cuda:1)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"  # works without `pip install -e .`

LAST_STEP="${LAST_STEP:-200}"
DIMS="${DIMS:-2 4 8 16}"
LAMBDAS="${LAMBDAS:-1 3 10 30 100 300 1000}"
DEV=${DEVICE:+--device $DEVICE}
CKPT=checkpoints/smoke/gaussian_lb
RES=results/smoke/gaussian_lb

run() { echo "+ $*"; if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi; }
[ "${DRY_RUN:-0}" = 1 ] || rm -rf "$CKPT" "$RES"

for n in $DIMS; do
  for lamb in $LAMBDAS; do
    lb="--dataset gaussian --data_dim $n --model mlp --hidden_units_per_dim 20 --num_hidden_layers 2
        --activation selu --lamb $lamb --batchsize 1024 --y_quick_topn 10 --y_steps 500
        --checkpoint_dir $CKPT"
    echo "=== [gaussian LB] train $LAST_STEP steps, n=$n, lambda=$lamb ==="
    run python train/train_lb.py -V $DEV $lb --num_Ck_samples 2 --y_init quick \
      --last_step "$LAST_STEP" --checkpoint_interval "$LAST_STEP" --lr 5.0e-4
    echo "=== [gaussian LB] eval (exhaustive, M=5), n=$n, lambda=$lamb ==="
    run python evaluation/eval_lb.py -V $DEV $lb --num_Ck_samples 5 --y_init exhaustive
  done
done

run python evaluation/plot_rdlb.py --checkpoint_dir $CKPT --out $RES/gaussian_lb.png
echo "=== gaussian LB smoke test PASSED (figure in $RES) ==="
