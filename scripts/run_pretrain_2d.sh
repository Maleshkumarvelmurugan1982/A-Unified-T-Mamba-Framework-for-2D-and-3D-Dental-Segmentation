#!/bin/bash
# STAGE 1 (2D): UniMiSS-2D self-supervised pretraining on X-rays (no labels). Optional: EXTRA_UNLABELED=/path/to/more/xrays
set -e
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
EXTRA=""
[ -n "${EXTRA_UNLABELED:-}" ] && EXTRA="--extra_unlabeled_dir $EXTRA_UNLABELED"
python pretrain_unimiss2d.py --dataset_path $DATA2D --out_dir $SSL2D --gpu $GPU \
   --epochs $SSL2D_EPOCHS --warmup_epochs 2 --samples_per_epoch 1000 --batch_size $SSL2D_BATCH --num_workers 8 \
   $EXTRA "$@" 2>&1 | tee -a $LOGS/ssl2d_pretrain_ep$SSL2D_EPOCHS.log
