#!/bin/bash
# Shared settings - sourced by every script. Override by exporting before running (e.g. GPU=1).
export BASE=${BASE:-/root/tmamba}
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONDA=$BASE/miniconda
export DATA=${DATA:-$BASE/datasets/NC-release-data-checked}
export SSL_EPOCHS=${SSL_EPOCHS:-30}   # pretraining epochs (single run: 30); each value gets its own output folder
export FT_EPOCHS=${FT_EPOCHS:-20}     # fine-tuning epochs (keep identical for baseline and UniMiSS)
export INIT_BEST=${INIT_BEST:-0}      # 1 = start SSL pretraining from your trained best_TMamba3D.pth (fast mode, see README)
export BEST=${BEST:-$REPO/pretrain/best_TMamba3D.pth}   # your saved baseline checkpoint (full checkpoint or bare state_dict)
export MODEL=${MODEL:-TMamba3D}   # TMamba3D (baseline / UniMiSS) or TMambaPFESA3D (PFESA)
SUFFIX=""; [ "$INIT_BEST" = "1" ] && SUFFIX="_fromBest"
export SSL=${SSL:-$BASE/runs_ssl/unimiss3d_ep${SSL_EPOCHS}${SUFFIX}}
export GPU=${GPU:-0}
export LOGS=$BASE/logs
export BATCH=${BATCH:-6}          # fine-tune / baseline batch size (lower it if you hit CUDA out-of-memory)
export SSL_BATCH=${SSL_BATCH:-3}  # pretraining batch size (each sample = 2 views; must be >= 2)
mkdir -p "$LOGS"

# ---------------- 2D (Tooth2D-X-Ray-6k) settings ----------------
export DATA2D=${DATA2D:-$BASE/datasets/Tooth2D-X-Ray-6k}
export SSL2D_EPOCHS=${SSL2D_EPOCHS:-30}     # UniMiSS-2D pretraining epochs (1000 random images each)
export SSL2D=${SSL2D:-$BASE/runs_ssl/unimiss2d_ep$SSL2D_EPOCHS}
export SSL2D_BATCH=${SSL2D_BATCH:-2}        # image PAIRS per pretraining step (network sees 2x as many images)
export FT2D_EPOCHS=${FT2D_EPOCHS:-30}       # fine-tuning epochs = original T-Mamba2D recipe
export BATCH2D=${BATCH2D:-4}                # original recipe batch size
export BEST2D=${BEST2D:-$REPO/pretrain/best_TMamba2D.pth}   # your reproduced baseline checkpoint (for testbase2d)

activate_env() {
  source $CONDA/etc/profile.d/conda.sh
  conda activate tmamba
  export CUDA_HOME=$CONDA_PREFIX
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  export LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
  mkdir -p /root/.cache/torch/kernels
  cd "$REPO"
  fix_triton_for_ada
}

# RTX 40-series (sm_89) and H100 (sm_90): Triton 2.0.x bundled with PyTorch 2.0.1 ships an old ptxas that rejects sm_89
# ("PTX .version 7.4 does not support .target sm_89"). Point Triton at the CUDA 11.8 ptxas from the conda env.
fix_triton_for_ada() {
  local cc
  cc=$(python -c "import torch;print('%d%d'%torch.cuda.get_device_capability(0))" 2>/dev/null || echo "")
  if [ "$cc" = "89" ] || [ "$cc" = "86" ] || [ "$cc" = "90" ] || [ "$cc" = "120" ]; then
    if [ -x "$CONDA_PREFIX/bin/ptxas" ]; then
      export TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas"
      local tdir bundled
      tdir=$(python -c "import triton,os;print(os.path.dirname(triton.__file__))" 2>/dev/null || echo "")
      bundled="$tdir/third_party/cuda/bin/ptxas"
      if [ -n "$tdir" ] && [ -e "$bundled" ] && [ ! -L "$bundled" ]; then
        cp -n "$bundled" "$bundled.orig" 2>/dev/null || true
        ln -sf "$CONDA_PREFIX/bin/ptxas" "$bundled" && echo "[fix] Triton now uses CUDA 11.8 ptxas for sm_$cc"
      fi
    else
      echo "[warn] $CONDA_PREFIX/bin/ptxas not found - Triton may fail on this GPU (sm_$cc)"
    fi
  fi
}
