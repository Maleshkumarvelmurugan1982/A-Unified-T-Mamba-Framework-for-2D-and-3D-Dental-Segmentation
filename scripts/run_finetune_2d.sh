#!/bin/bash
# STAGE 2 (2D): fine-tune T-Mamba2D from the UniMiSS-2D encoder with the ORIGINAL recipe (30 epochs, lr 0.0075, batch 4).
set -e
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
python ./train.py --dataset Tooth2D-X-Ray-6k --dimension 2d --model TMamba2D --scaling_version SMALL \
  --epoch $FT2D_EPOCHS --multi_gpu False --dataset_path $DATA2D --gpu $GPU --batch_size $BATCH2D \
  --pretrain_weight ${FT_INIT:-$SSL2D/unimiss_encoder_teacher.pth} ${LR:+--lr $LR} \
  2>&1 | tee $LOGS/finetune2d_unimiss_ep$SSL2D_EPOCHS.log
