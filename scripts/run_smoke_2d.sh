#!/bin/bash
# 1-epoch / 8-image UniMiSS-2D smoke test. Passing = pipeline works on your GPU.
set -e
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
python pretrain_unimiss2d.py --dataset_path $DATA2D --out_dir $BASE/runs_ssl/smoke2d --gpu $GPU \
   --epochs 1 --samples_per_epoch 8 --batch_size 2 --num_workers 2 --warmup_epochs 0
ls -la $BASE/runs_ssl/smoke2d && echo "SMOKE TEST 2D PASSED"
