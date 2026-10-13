#!/bin/bash
# 2D ONE-COMMAND LAUNCHER (T-Mamba2D + UniMiSS-2D pretraining).   Usage:
#   bash launch2d.sh            # setup -> data -> smoke -> pretrain -> finetune -> test
#   bash launch2d.sh <step>     # setup | data | smoke | pretrain | finetune | test | testbase | baseline
# Recommended:  nohup bash launch2d.sh > launch2d.log 2>&1 &    then    tail -f launch2d.log
set -e
cd "$(dirname "$0")"
S=scripts
step() {
  case "$1" in
    setup)    bash $S/setup_env.sh ;;
    data)     bash $S/download_data_2d.sh ;;
    smoke)    bash $S/run_smoke_2d.sh ;;
    pretrain) bash $S/run_pretrain_2d.sh ;;
    finetune) bash $S/run_finetune_2d.sh ;;
    test)     bash $S/run_test_2d.sh ;;
    testbase) bash $S/run_test_2d.sh "${BEST2D:-$PWD/pretrain/best_TMamba2D.pth}" ;;
    baseline) bash $S/run_baseline_2d.sh; bash $S/run_test_2d.sh ;;
    *) echo "unknown step $1"; exit 1 ;;
  esac
}
if [ -z "$1" ]; then
  for s in setup data smoke pretrain finetune test; do echo; echo "################ $s ################"; step $s; done
  echo; echo "ALL DONE. Checkpoint: ./runs/*/checkpoints/best_TMamba2D.pth"
else
  step "$1"
fi
