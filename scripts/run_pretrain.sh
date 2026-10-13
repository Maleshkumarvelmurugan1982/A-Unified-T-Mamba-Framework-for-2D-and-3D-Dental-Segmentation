#!/bin/bash
# STAGE 1: UniMiSS-3D self-supervised pretraining (volume-slice consistency self-distillation, no labels).
set -e
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
# Add "--init_weight $REPO/pretrain/best_TMamba3D.pth" to start from your checkpoint (not recommended for the cleanest paper comparison).
# If interrupted, resume with: bash scripts/run_pretrain.sh --resume $SSL/last.pth
EXTRA=""
if [ "$INIT_BEST" = "1" ]; then EXTRA="--init_weight $BEST"; echo "[info] starting SSL from $BEST"; fi
python pretrain_unimiss3d.py --dataset_path $DATA --out_dir $SSL --gpu $GPU \
   --epochs $SSL_EPOCHS --warmup_epochs 2 --samples_per_epoch 256 --batch_size $SSL_BATCH --num_workers 12 $EXTRA "$@" 2>&1 | tee -a $LOGS/ssl_pretrain_ep$SSL_EPOCHS.log
