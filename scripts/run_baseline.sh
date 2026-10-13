#!/bin/bash
# Baseline T-Mamba from scratch, identical settings (for the paper comparison table).
set -e
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
python ./train.py --dataset 3D-CBCT-Tooth --model $MODEL --dimension 3d \
  --dataset_path $DATA --gpu $GPU --scaling_version TINY --epoch $FT_EPOCHS --batch_size $BATCH --multi_gpu False 2>&1 | tee $LOGS/finetune_baseline.log
