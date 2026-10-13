<p align="center"><img src="images/T-Mamba-logo.png" width="130"></p>

# A Unified T-Mamba Framework for 2D and 3D Dental Segmentation

Final-year project (CSE). Guide: Mrs. T. Bhuvaneswari. Team: V. Maleshkumar, S. Sujan, R. Mallesh Kumar.

This repository is built on the official **T-Mamba** code ([paper](https://arxiv.org/pdf/2404.01065), MIT licence, original authors: yykzjh / Bryceee et al.).
The original model code is unchanged. We add one novelty for each dimension:

| | Baseline (T-Mamba) | Our novelty | Where |
|---|---|---|---|
| **2D** panoramic X-ray | T-Mamba2D, trained from scratch | **UniMiSS-2D**: label-free self-distillation pre-training of the encoder, then the original fine-tune recipe | `lib/ssl/unimiss2d.py`, `lib/ssl/dataset2d.py`, `pretrain_unimiss2d.py` |
| **3D** CBCT | T-Mamba3D | **PFESA**: parameter-free frequency + spatial attention on the 3 skip connections | `lib/models/pfesa.py`, `lib/models/TMambaPFESA3D.py` |

The architecture is not enlarged: UniMiSS-2D changes only the initialisation, and PFESA adds **0 parameters** (weights stay compatible with the baseline).

## Results (our own runs, same data and setup for baseline and novelty)

**2D** (Tooth2D-X-Ray-6k, 5,000 train / 1,225 test, 640x1280)

| Model | DSC | IoU | ACC | JI |
|---|---|---|---|---|
| T-Mamba2D (baseline) | 0.884233 | 0.797599 | 0.961985 | 0.796064 |
| + UniMiSS-2D | **0.887013** | **0.802213** | **0.962939** | **0.800753** |

**3D** (CBCT, foreground row, 1.31 M params for both)

| Model | DSC | IoU | SO | HD | ASSD |
|---|---|---|---|---|---|
| T-Mamba3D (baseline) | 93.45 | 88.03 | 97.06 | 7.838 | 0.818 |
| + PFESA | **93.47** | **88.08** | **97.37** | **1.131** | **0.398** |

The gain in 2D is small and comes from a single run. In 3D the overlap metrics barely move, while the boundary metrics (HD, ASSD) improve clearly.
Absolute baseline numbers differ slightly from the published paper (different batch size and GPU setup); compare novelty against our baseline, not against the paper.
Plots are in `docs/images/`.

## Repository layout

```
lib/models/TMamba2D.py, TMamba3D.py     original T-Mamba (2D has forward_features() for SSL)
lib/models/pfesa.py                     PFESA module (3D)                          [novelty]
lib/models/TMambaPFESA3D.py             T-Mamba3D + PFESA on the 3 skips           [novelty]
lib/ssl/unimiss2d.py, dataset2d.py      UniMiSS-2D student/teacher pre-training    [novelty]
lib/ssl/unimiss3d.py, dataset.py        UniMiSS 3D variant (experiment)
pretrain_unimiss2d.py                   stage 1 for 2D
train.py, test.py, inference.py         train / evaluate / infer (2D and 3D)
Tim/                                    causal-conv1d 1.0.0 and mamba_ssm 1.0.1 sources (built locally)
scripts/                                one-command pipelines (setup, data, smoke, pretrain, finetune, test, pfesa)
launch2d.sh / launch.sh                 2D launcher / 3D launcher
demo_infer_2d.py, demo_web.py           image in -> mask out (CLI and small upload page)
demo/                                   Kaggle notebook (2D + 3D demo) and run3d.py
tools/                                  metrics, FLOPs, PFESA check, infer_visualize.py
docs/                                   method notes, original README, result plots
```

## Setup

```bash
conda create -n tmamba python=3.9 && conda activate tmamba
pip install -r requirements.txt
cd Tim/causal-conv1d && python setup.py install && cd ../mamba && python setup.py install && cd ../..
```
Needs an NVIDIA GPU, CUDA 11.8 and PyTorch 2.0.1 (`mamba_ssm==1.0.1`, `causal_conv1d==1.0.0`). `scripts/setup_env.sh` automates this on a fresh VM.

## 2D: baseline vs UniMiSS-2D

```bash
bash launch2d.sh setup      # environment
bash launch2d.sh data       # Tooth2D-X-Ray-6k
bash launch2d.sh smoke      # 1-epoch check
bash launch2d.sh baseline   # baseline (30 epochs) + test
bash launch2d.sh pretrain   # UniMiSS-2D stage 1 (label-free)
bash launch2d.sh finetune   # stage 2: original recipe, initialised from stage 1
bash launch2d.sh test       # DSC / IoU / ACC / JI
```
Details: `docs/README_2D_UNIMISS.md`.

## 3D: baseline vs PFESA

```bash
bash launch.sh setup
bash launch.sh data         # CBCT dataset + baseline checkpoint
bash launch.sh baseline     # T-Mamba3D train + test
bash launch.sh pfesacheck   # PFESA sanity check
bash launch.sh pfesa        # T-Mamba3D+PFESA train + test (resumes after an interruption)
```
Details: `docs/README_UNIMISS3D.md`, `tools/check_pfesa.py`.

## Live demo

```bash
python demo_infer_2d.py --weight best_TMamba2D.pth --input my_xray.png --out demo_out
python tools/infer_visualize.py --data <CBCT folder> --base best_TMamba3D.pth --pfesa best_TMambaPFESA3D.pth --out viz --n 2
```
`demo/T-Mamba_2D_3D_Full_Demo_Kaggle.ipynb` builds the whole environment on Kaggle (GPU + Internet on) and runs both demos.
Note: this notebook was assembled from cells that each ran separately; it has not yet been run top to bottom as one notebook.

## Important implementation note

`SpectralGatingBlocks`, `GateModules` and the positional embeddings are stored in plain Python lists, so they are **not part of the checkpoint**.
They are re-created from the random seed, so `seed = 1777777` must be set immediately before building the model (as `test.py` does). Without this, outputs will not match the trained model.

## Datasets: download and folder setup

One command per dataset (needs `pip install gdown` and Internet). Files go straight into the right folders inside the repo:

```bash
pip install gdown
python tools/download_data.py --what 2d      # 2D X-ray dataset
python tools/download_data.py --what 3d      # 3D CBCT dataset
python tools/download_data.py --what ckpt3d  # 3D baseline checkpoint (optional)
python tools/download_data.py --what all     # everything
```

| Dataset | Google Drive file id(s) | Ends up in |
|---|---|---|
| 2D Tooth2D-X-Ray-6k (2 parts) | `1YTa8Z5iv_pO0G-0KeU-yTritkodg7TMK`, `1CUq-MuMIr7eHKBVZ4VTh-m72pcLMhgp2` | `datasets/Tooth2D-X-Ray-6k/` |
| 3D CBCT (NC-release-data-checked) | `1ivIhA222AoFxzTYWVAC3yUFSsg5Waqqu` | `datasets/NC-release-data-checked/` |
| 3D baseline checkpoint | `1WQeyZLiUeiNm8onicyeB5-TipOPZuaHu` | `pretrain/best_TMamba3D.pth` |

Link format: `https://drive.google.com/file/d/<id>/view`. If Drive shows a quota error, download in the browser, save into `datasets/_downloads/` as `2d_part1`, `2d_part2`, `3d_data` (or `3d_ckpt`), and run the command again.

Expected layout after download:

```
datasets/
  Tooth2D-X-Ray-6k/            train/{images,annotations}   test/{images,annotations}     (5,000 train / 1,225 test)
  NC-release-data-checked/     train/{images,labels}        valid/{images,labels}         (*.nii.gz)
```

Training and testing with these folders directly (without the launch scripts):

```bash
# 2D baseline (30 epochs, batch 4 = original recipe)
python train.py --dataset Tooth2D-X-Ray-6k --dimension 2d --model TMamba2D --scaling_version SMALL \
  --epoch 30 --multi_gpu False --batch_size 4 --dataset_path ./datasets/Tooth2D-X-Ray-6k
python test.py  --dataset Tooth2D-X-Ray-6k --dimension 2d --model TMamba2D --scaling_version SMALL \
  --dataset_path ./datasets/Tooth2D-X-Ray-6k --pretrain_weight pretrain/best_TMamba2D.pth

# 3D baseline and PFESA (20 epochs)
python train.py --dataset 3D-CBCT-Tooth --dimension 3d --model TMambaPFESA3D --scaling_version TINY \
  --epoch 20 --multi_gpu False --batch_size 6 --dataset_path ./datasets/NC-release-data-checked
python test.py  --dataset 3D-CBCT-Tooth --dimension 3d --model TMambaPFESA3D --scaling_version TINY \
  --dataset_path ./datasets/NC-release-data-checked --pretrain_weight best_TMambaPFESA3D.pth
```
(For the 3D baseline use `--model TMamba3D`.)

The `launch.sh` / `launch2d.sh` scripts use `/root/tmamba/datasets` by default; set `BASE=<folder>` or `DATA=` / `DATA2D=` (e.g. `DATA2D=$PWD/datasets/Tooth2D-X-Ray-6k`) to point them at the repo folders.
The public TED dataset of the original authors is at [Hugging Face](https://huggingface.co/datasets/Bryceee/TED).

## Weights

`pretrain/best_TMamba2D.pth` (our 2D baseline, 4 MB) is included so the 2D demo works right after cloning.
The 3D checkpoints are larger: put `best_TMambaPFESA3D.pth` in GitHub Releases and link it here.

## Credits and licence

Base code: T-Mamba by the original authors (MIT, see `LICENSE`). UniMiSS idea: Xie et al., ECCV 2022. PFESA idea: Li et al., MICCAI 2025.
Our additions are released under the same MIT licence.
