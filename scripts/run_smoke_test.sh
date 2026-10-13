#!/bin/bash
# 1-epoch / 8-sample SSL smoke test (a few minutes). Passing = the pipeline works on your GPU.
set -e
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
python pretrain_unimiss3d.py --dataset_path $DATA --out_dir $BASE/runs_ssl/smoke --gpu $GPU \
   --epochs 1 --samples_per_epoch 8 --batch_size 2 --num_workers 2 --warmup_epochs 0
ls -la $BASE/runs_ssl/smoke && echo "SMOKE TEST PASSED"
