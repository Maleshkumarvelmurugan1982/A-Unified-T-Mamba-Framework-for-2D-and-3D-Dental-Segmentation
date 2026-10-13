#!/bin/bash
# Baseline T-Mamba2D from scratch, original recipe (only needed if you want to re-run the baseline).
set -e
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
python ./train.py --dataset Tooth2D-X-Ray-6k --dimension 2d --model TMamba2D --scaling_version SMALL \
  --epoch $FT2D_EPOCHS --multi_gpu False --dataset_path $DATA2D --gpu $GPU --batch_size $BATCH2D \
  2>&1 | tee $LOGS/finetune2d_baseline.log
