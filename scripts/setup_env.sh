#!/bin/bash
# One-time environment setup (Miniconda + PyTorch 2.0.1 cu118 + mamba-ssm build). Safe to re-run.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [ -f $BASE/.env_ready ]; then echo "Environment already set up ($BASE/.env_ready). Delete that file to force a rebuild."; exit 0; fi

echo; echo "######## 1/8 sanity check ########"
(
set -e
echo '===== SYSTEM ====='
python3 --version
uname -a
df -h /root
echo
echo '===== NVIDIA GPU ====='
nvidia-smi
echo
echo '===== CUDA TOOLKIT ====='
which nvcc || true
nvcc --version 2>/dev/null || true

)

echo; echo "######## 2/8 miniconda + tmamba env ########"
(
set -e
BASE='/root/tmamba'
CONDA='/root/tmamba/miniconda'
mkdir -p '/root/tmamba'
cd '/root/tmamba'

if [ ! -x "$CONDA/bin/conda" ]; then
    echo '===== DOWNLOADING MINICONDA ====='
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$CONDA"
    rm -f miniconda.sh
else
    echo 'Miniconda already installed.'
fi

"$CONDA/bin/conda" --version
source "$CONDA/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if conda env list | grep -q "$CONDA/envs/tmamba"; then
    echo 'tmamba env already exists.'
else
    echo '===== CREATING tmamba ENV (Python 3.9) ====='
    conda create -y -n tmamba python=3.9 pip
fi

conda activate tmamba
echo
echo '===== ENVIRONMENT ====='
which python
python --version

)

echo; echo "######## 3/8 PyTorch 2.0.1+cu118 ########"
(
set -e
source /root/tmamba/miniconda/etc/profile.d/conda.sh
conda activate tmamba

python -m pip install --upgrade pip
pip install \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('PyTorch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print('GPU', i, ':', torch.cuda.get_device_name(i), '| capability:', torch.cuda.get_device_capability(i))
assert torch.cuda.is_available()
    
PY

)

echo; echo "######## 4/8 CUDA 11.8 toolkit + compilers ########"
(
set -e
source /root/tmamba/miniconda/etc/profile.d/conda.sh
conda activate tmamba

echo '===== INSTALLING CUDA 11.8 DEV TOOLKIT ====='
conda install -y -c nvidia/label/cuda-11.8.0 \
    cuda-nvcc \
    cuda-cudart-dev \
    libcublas-dev \
    libcusparse-dev \
    libcusolver-dev \
    libcurand-dev \
    libcufft-dev \
    libnvjpeg-dev \
    cuda-cccl

echo '===== INSTALLING MATCHING COMPILERS ====='
conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH

echo
echo '===== VERIFY ====='
which nvcc
nvcc --version
echo "CUDA_HOME=$CUDA_HOME"

)

echo; echo "######## 5/8 python dependencies ########"
(
set -e
source /root/tmamba/miniconda/etc/profile.d/conda.sh
conda activate tmamba

python -m pip install --no-cache-dir 'setuptools==69.5.1' ninja packaging

pip install --no-cache-dir \
    batchgenerators==0.25 \
    einops==0.7.0 \
    matplotlib==3.8.4 \
    matplotlib-inline==0.1.6 \
    ipython==8.18.1 \
    ml_collections==0.1.1 \
    monai==1.3.0 \
    nibabel==5.2.0 \
    nni==3.0 \
    numpy==1.24.1 \
    opencv-python==4.8.1.78 \
    Pillow==10.2.0 \
    scipy==1.11.4 \
    SimpleITK==2.3.1 \
    scikit-image \
    tifffile==2023.7.10 \
    timm==0.9.12 \
    torchtoolbox==0.1.8.2 \
    tqdm==4.65.0 \
    ptflops \
    gdown

pip install --no-cache-dir Surface_Distance_Based_Measures==0.1 || \
    echo 'WARNING: Surface_Distance_Based_Measures failed to install — HD/ASSD metrics may be unavailable.'

pip uninstall -y transformers tokenizers -q || true
pip install --no-cache-dir transformers==4.30.2 tokenizers==0.13.3

python - <<'PY'
import batchgenerators, einops, matplotlib, ml_collections, monai, nibabel
import cv2, PIL, scipy, SimpleITK, skimage, tifffile, timm, tqdm, transformers
import IPython, matplotlib_inline
import matplotlib.pyplot as plt
print('All core deps import OK')
print('PyTorch:', __import__('torch').__version__)
print('transformers:', transformers.__version__)
print('matplotlib:', matplotlib.__version__)
print('IPython:', IPython.__version__)
print('matplotlib-inline:', matplotlib_inline.__version__)
PY

)

echo; echo "######## 6/8 build causal-conv1d + mamba-ssm ########"
(
set -e
set -o pipefail
source /root/tmamba/miniconda/etc/profile.d/conda.sh
conda activate tmamba

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CPATH=$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH

NPROC=$(nproc)
if [ "$NPROC" -gt 12 ]; then MAX_JOBS=12; else MAX_JOBS=$NPROC; fi
export MAX_JOBS
echo "nproc=$NPROC -> using MAX_JOBS=$MAX_JOBS"

ARCH=$(python -c "import torch; maj,min_=torch.cuda.get_device_capability(0); print(f'{maj}.{min_}')")
export TORCH_CUDA_ARCH_LIST="$ARCH"
echo "Detected GPU compute capability -> TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (A100 should be 8.0)"

mkdir -p /root/tmamba/build_logs
cd $REPO/Tim/causal-conv1d
echo '===== BUILDING causal-conv1d ====='
pip uninstall -y causal_conv1d 2>/dev/null || true
pip install -e . --no-build-isolation --no-cache-dir -v 2>&1 | tee /root/tmamba/build_logs/causal_conv1d.log | tail -n 20

cd $REPO/Tim/mamba
echo '===== BUILDING mamba-ssm ====='
pip uninstall -y mamba_ssm 2>/dev/null || true
pip install -e . --no-build-isolation --no-cache-dir -v 2>&1 | tee /root/tmamba/build_logs/mamba_ssm.log | tail -n 20

echo
echo '===== VERIFY ====='
cd /tmp
python - <<'PY'
import torch, causal_conv1d, mamba_ssm
from mamba_ssm import Mamba, Mamba_FFT
print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0), '| capability:', torch.cuda.get_device_capability(0))
print('causal_conv1d:', causal_conv1d.__version__)
print('mamba_ssm: OK')
PY

)

echo; echo "######## 7/8 free disk space ########"
(
set -e
source /root/tmamba/miniconda/etc/profile.d/conda.sh
conda activate tmamba

echo '===== BEFORE CLEANUP ====='
df -h /root/tmamba
conda clean --all -y
pip cache purge
rm -rf $REPO/Tim/mamba/build
rm -rf $REPO/Tim/causal-conv1d/build
find $REPO -type d -name '__pycache__' -prune -exec rm -rf {} +
echo
echo '===== AFTER CLEANUP ====='
df -h /root/tmamba

)

echo; echo "######## 8/8 final check ########"
(
source $CONDA/etc/profile.d/conda.sh; conda activate tmamba
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
cd "$REPO"
python - <<'PY'
import re, torch
from pathlib import Path
import torchtoolbox, os
# torchtoolbox NumPy>=1.24 compatibility
p = Path(os.path.dirname(torchtoolbox.__file__)) / "transform" / "autoaugment.py"
if p.exists():
    s = p.read_text(); s2 = re.sub(r"np\.int\b","int",s); s2 = re.sub(r"np\.float\b","float",s2); s2 = re.sub(r"np\.bool\b","bool",s2)
    if s2 != s: p.write_text(s2); print("patched torchtoolbox")
print("PyTorch", torch.__version__, "| CUDA", torch.version.cuda, "| available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA GPU visible"
print("GPU:", torch.cuda.get_device_name(0))
from mamba_ssm import Mamba, Mamba_FFT
import causal_conv1d
print("mamba_ssm + causal_conv1d OK")
import train
print("train.py imports OK")
PY
)
touch $BASE/.env_ready
echo; echo "ENVIRONMENT READY"
