#!/bin/bash
# Evaluate on the test split -> valid_DSC / valid_IoU / valid_ACC / valid_JI (same metrics as your baseline table).
set -e
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
# usage: bash scripts/run_test_2d.sh [path/to/best_TMamba2D.pth]   (default: newest run under ./runs)
CK=${1:-$(ls -dt runs/*TMamba2D*Tooth2D* | head -1)/checkpoints/best_TMamba2D.pth}
echo "Testing $CK"
python ./test.py --dataset Tooth2D-X-Ray-6k --model TMamba2D --dimension 2d --scaling_version SMALL \
  --dataset_path $DATA2D --gpu $GPU --pretrain_weight "$CK" 2>&1 | tee -a $LOGS/test2d.log
