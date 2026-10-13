#!/bin/bash
# STAGE 2: fine-tune T-Mamba from the UniMiSS-3D encoder (20 epochs, batch 6 - same as baseline).
set -e
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
python ./train.py --dataset 3D-CBCT-Tooth --model $MODEL --dimension 3d \
  --dataset_path $DATA --gpu $GPU \
  --pretrain_weight ${FT_INIT:-$SSL/unimiss_encoder_teacher.pth} \
  --scaling_version TINY --epoch $FT_EPOCHS --batch_size $BATCH --multi_gpu False 2>&1 | tee $LOGS/finetune_unimiss_ep$SSL_EPOCHS.log
