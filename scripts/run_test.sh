#!/bin/bash
# STAGE 3: evaluate -> HD / ASSD / IoU / SO / DSC table.
set -e
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
# usage: bash scripts/run_test.sh [path/to/best_TMamba3D.pth]   (default: newest run under ./runs)
CK=${1:-$(ls -dt runs/*_${MODEL}_3D-CBCT-Tooth | head -1)/checkpoints/best_${MODEL}.pth}
echo "Testing $CK"
python ./test.py --dataset 3D-CBCT-Tooth --model $MODEL --dimension 3d --scaling_version TINY \
  --dataset_path $DATA --gpu $GPU --pretrain_weight "$CK" 2>&1 | tee -a $LOGS/test.log
