#!/bin/bash
# PFESA training, SPOT-SAFE: if the instance was shut down mid-training, just run `bash launch.sh pfesa` again -
# it finds the newest PFESA run, and continues from its last finished epoch (checkpoint is saved every epoch).
# Default = from scratch, identical settings to the baseline. Warm start: FT_INIT=$BEST bash launch.sh pfesa
set -e
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export MODEL=TMambaPFESA3D
activate_env
LAST_DIR=$(ls -dt runs/*_${MODEL}_3D-CBCT-Tooth 2>/dev/null | head -1 || true)
EXTRA=""
if [ -n "$LAST_DIR" ] && [ -f "$LAST_DIR/checkpoints/latest_${MODEL}.state" ] && [ -f "$LAST_DIR/checkpoints/latest_${MODEL}.pth" ]; then
  if [ -f "$LAST_DIR/checkpoints/.finished" ]; then
    echo "Previous PFESA run in $LAST_DIR already finished - starting a NEW run. (delete runs/ folder to avoid this)"
  else
    echo "RESUMING interrupted run: $LAST_DIR"
    EXTRA="--resume $LAST_DIR/checkpoints/latest_${MODEL}.state --pretrain_weight $LAST_DIR/checkpoints/latest_${MODEL}.pth"
  fi
elif [ -n "$FT_INIT" ]; then
  EXTRA="--pretrain_weight $FT_INIT"
fi
python ./train.py --dataset 3D-CBCT-Tooth --model $MODEL --dimension 3d \
  --dataset_path $DATA --gpu $GPU --scaling_version TINY --epoch $FT_EPOCHS --batch_size $BATCH --multi_gpu False $EXTRA 2>&1 | tee -a $LOGS/pfesa.log
LAST_DIR=$(ls -dt runs/*_${MODEL}_3D-CBCT-Tooth | head -1)
touch "$LAST_DIR/checkpoints/.finished"
bash "$(dirname "${BASH_SOURCE[0]}")/run_test.sh"
