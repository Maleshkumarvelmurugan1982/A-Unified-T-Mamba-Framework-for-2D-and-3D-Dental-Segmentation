#!/bin/bash
# ONE-COMMAND LAUNCHER (PFESA edition: no argument = setup -> data -> pfesacheck -> pfesa train+test).   Usage:
#   bash launch.sh            # full pipeline: env -> data -> smoke test -> SSL pretrain -> fine-tune -> test
#   bash launch.sh baseline   # then run the from-scratch baseline + test it (for the comparison table)
#   bash launch.sh testbase   # evaluate YOUR saved best_TMamba3D.pth (baseline numbers, no training)
#   bash launch.sh pfesacheck # PFESA GPU sanity check;  bash launch.sh pfesa  # train+test TMambaPFESA3D
#   SPOT SAFE: if the instance gets shut down, restart it and run   nohup bash launch.sh pfesa > launch2.log 2>&1 &   (it resumes)
#   bash launch.sh <step>     # one step: setup | data | smoke | pretrain | finetune | test | testbase  | baseline
# Recommended (keeps running after you disconnect):   nohup bash launch.sh > launch.log 2>&1 &   then   tail -f launch.log
set -e
cd "$(dirname "$0")"
# RunPod: keep everything on the persistent /workspace volume (container disk is small and is wiped on restart)
if [ -d /workspace ] && [ ! -e /root/tmamba ]; then mkdir -p /workspace/tmamba && ln -s /workspace/tmamba /root/tmamba; fi
S=scripts
step() {
  case "$1" in
    setup)    bash $S/setup_env.sh ;;
    data)     bash $S/download_data.sh ;;
    smoke)    bash $S/run_smoke_test.sh ;;
    pretrain) bash $S/run_pretrain.sh ;;
    finetune) bash $S/run_finetune.sh ;;
    test)     bash $S/run_test.sh ;;
    testbase) MODEL=TMamba3D bash $S/run_test.sh "${BEST:-$PWD/pretrain/best_TMamba3D.pth}" ;;
    pfesacheck) bash $S/run_pfesa_check.sh ;;
    pfesa)    bash $S/run_pfesa.sh ;;
    baseline) MODEL=TMamba3D bash $S/run_baseline.sh; MODEL=TMamba3D bash $S/run_test.sh ;;
    *) echo "unknown step $1"; exit 1 ;;
  esac
}
if [ -z "$1" ]; then
  for s in setup data pfesacheck pfesa; do echo; echo "################ $s ################"; step $s; done
  echo; echo "ALL DONE. PFESA checkpoint: ./runs/*_TMambaPFESA3D_*/checkpoints/best_TMambaPFESA3D.pth"
else
  step "$1"
fi
