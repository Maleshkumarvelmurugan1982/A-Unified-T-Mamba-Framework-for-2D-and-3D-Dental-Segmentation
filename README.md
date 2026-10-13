<p align="center"><img src="images/T-Mamba-logo.png" width="130"></p>

<h1 align="center">A Unified T-Mamba Framework for 2D and 3D Dental Segmentation</h1>

<p align="center">
UniMiSS-2D pre-training for panoramic X-rays &nbsp;|&nbsp; PFESA skip-attention for CBCT volumes<br>
Final-year project, Computer Science and Engineering
</p>

**Repository:** https://github.com/Maleshkumarvelmurugan1982/A-Unified-T-Mamba-Framework-for-2D-and-3D-Dental-Segmentation
**Guide:** Mrs. T. Bhuvaneswari &nbsp;·&nbsp; **Team:** V. Maleshkumar, S. Sujan, R. Mallesh Kumar

---

## Contents
1. [What this project is](#1-what-this-project-is)
2. [What we did, step by step](#2-what-we-did-step-by-step)
3. [Results](#3-results)
4. [Method details](#4-method-details)
5. [Repository layout](#5-repository-layout)
6. [Before you start: what computer do you need?](#6-before-you-start-what-computer-do-you-need)
7. [Setup on your own machine (step by step)](#7-setup-on-your-own-machine-step-by-step)
8. [Setup without a GPU: run on Kaggle](#8-setup-without-a-gpu-run-on-kaggle)
9. [Datasets: download and folder setup](#9-datasets-download-and-folder-setup)
10. [Check that everything works](#10-check-that-everything-works)
11. [Run the 2D experiments](#11-run-the-2d-experiments)
12. [Run the 3D experiments](#12-run-the-3d-experiments)
13. [Live demo: image in, mask out](#13-live-demo-image-in-mask-out)
14. [Evaluation metrics](#14-evaluation-metrics)
15. [Important implementation notes](#15-important-implementation-notes)
16. [Troubleshooting](#16-troubleshooting)
17. [Weights](#17-weights)
18. [Credits and licence](#18-credits-and-licence)

---

## 1. What this project is

Teeth have to be outlined (segmented) in dental images before many diagnosis and planning steps. We work with two image types:

- **2D panoramic X-rays** (one flat image, 640x1280 after resizing)
- **3D CBCT scans** (a volume of slices)

**T-Mamba** ([paper](https://arxiv.org/pdf/2404.01065)) is a network for exactly this task. It uses Mamba (a state-space sequence model) to capture long-range context in two domains: the normal image space and the frequency (FFT) space. The same network family handles 2D and 3D.

We started from the official T-Mamba code, reproduced its baseline, and added **one improvement for each dimension**:

| | Baseline | Our novelty | Idea in one line | Code |
|---|---|---|---|---|
| **2D** | T-Mamba2D from scratch | **UniMiSS-2D** | Pre-train the encoder on X-rays *without labels*, then fine-tune. Better starting weights, same network. | `pretrain_unimiss2d.py`, `lib/ssl/unimiss2d.py`, `lib/ssl/dataset2d.py` |
| **3D** | T-Mamba3D | **PFESA** | Add a parameter-free attention (edges from high frequencies, regions from low frequencies) on the 3 skip connections. | `lib/models/pfesa.py`, `lib/models/TMambaPFESA3D.py` |

Neither change adds parameters: 2D has 0.943 M and 3D has 1.31 M, exactly like the baselines.

## 2. What we did, step by step

1. **Reproduced the baselines.** Trained T-Mamba2D on the 2D X-ray dataset and T-Mamba3D on the CBCT dataset with the original recipes, and measured them ourselves.
2. **2D novelty (UniMiSS-2D).** Wrote label-free student/teacher self-distillation pre-training for the T-Mamba2D encoder, then fine-tuned with the *unchanged* original recipe starting from the pre-trained encoder.
3. **3D novelty (PFESA).** Added a frequency-plus-spatial attention module with no learnable parameters on the 3 skip connections, so baseline weights still load.
4. **Comparison with the base paper.** The values of our novelties are compared with the published T-Mamba paper results on the same test sets (2D: 1,225 X-rays, 3D: 26 CBCT volumes).
5. **Live demo.** Built a one-image inference script and a Kaggle notebook that shows the segmented mask (2D) and baseline vs PFESA (3D).
6. **Packaged everything** into this repository with one-command scripts and dataset download tools.

## 3. Results

The values of our novelties are compared with the **published T-Mamba (base paper) results**. All "proposed" numbers come from our own runs on the test splits (2D: 1,225 X-rays, 3D: 26 CBCT volumes).

### 3.1 2D: T-Mamba paper vs UniMiSS-2D (proposed)

| Metric | T-Mamba paper | UniMiSS-2D Self-Distillation (proposed) |
|---|---|---|
| IoU (%) ↑ | 81.52 | 80.22 |
| DSC (%) ↑ | 88.25 | **88.70** |
| ACC (%) ↑ | 96.55 | 96.29 |
| JI (%) ↑ | – | 80.07 |
| Parameters (M) ↓ | 0.68 | 0.943 |
| FLOPs (G) ↓ | 19.58 | **18.744** |

- **Better than the paper:** DSC (+0.45 points) and FLOPs (18.744 G vs 19.58 G, about 4% lower).
- **Slightly below the paper:** IoU (-1.30 points) and ACC (-0.26 points).
- **Larger model:** 0.943 M parameters vs 0.68 M in the paper.
- JI is not reported in the paper, so there is nothing to compare it with.
- The result comes from a single run; more seeds would be needed to call small differences statistically solid.

### 3.2 3D: T-Mamba paper vs PFESA (proposed)

| Metric | T-Mamba paper | PFESA (proposed) |
|---|---|---|
| DSC (%) ↑ | 93.60 | **96.70** |
| IoU (%) ↑ | 88.31 | **93.98** |
| SO (%) ↑ | 97.53 | **98.53** |
| HD ↓ (paper: mm, ours: voxels) | **0.0118** | 0.5954 |
| ASSD ↓ (paper: mm, ours: voxels) | 0.42 | **0.21** |

Parameters: 1.31 M for both (PFESA adds 0 parameters). FLOPs: paper 50.56 G, PFESA about 51.9 G (analytic estimate, not profiler-measured).

- **Better than the paper:** DSC (+3.10), IoU (+5.67), SO (+1.00) and ASSD (0.21 vs 0.42, lower is better).
- **Worse than the paper:** HD (0.5954 vs 0.0118).
- **Units:** the paper reports HD and ASSD in **millimetres** (Table III of the paper); our HD and ASSD are measured in **voxels** (volumes resampled to 0.5 mm), so these two rows are not directly comparable and should be read with care.

**How our 3D values are computed.** The evaluation code prints one row per class (background, foreground/tooth) and an `average` row over both classes. The PFESA values in the table above are the `average` row of our evaluation run.

The paper trained on 4 GPUs with batch 16; we used a single GPU with a smaller batch, and training is not bit-exact reproducible, so small differences from the published values are expected.

Plots are in [`docs/images/`](docs/images).

## 4. Method details

### 4.1 UniMiSS-2D (2D)
A 2D adaptation of the UniMiSS self-supervised idea (Xie et al., ECCV 2022). No labels are used and the test split is never touched.

1. **Stage 1, pre-training** (`pretrain_unimiss2d.py`): a student and an EMA teacher see different augmented views of the same X-ray (one global view plus 4 regional strips). An 8-term distillation loss pulls them together. The teacher is an exponential moving average of the student (momentum 0.996 rising to 1) with centring and sharpening. Each epoch uses 1,000 random training images, 30 epochs.
2. **Stage 2, fine-tuning**: the original T-Mamba2D recipe (30 epochs, AdamW, lr 0.0075, MultiStepLR milestones [24, 28], Dice loss), starting from the Stage 1 encoder.
3. **Stage 3, testing**: valid DSC, IoU, ACC, JI.

Not ported to 2D: UniMiSS' switchable 2D/3D embedding and X-ray + CT mixing, because the T-Mamba encoder is not dimension-free. More: [`docs/README_2D_UNIMISS.md`](docs/README_2D_UNIMISS.md).

### 4.2 PFESA (3D)
PFESA (Li et al., MICCAI 2025), adapted to 3D feature maps, applied to the 3 skip connections of T-Mamba3D (`lib/models/pfesa.py`). For a skip feature `X`:

- Split `X` into low- and high-frequency parts with an FFT and a Gaussian low-pass `G` (`sigma = r * min(dims) / max(dims)`, `r = 0.1`), giving `X_l` and `X_h`.
- Edge attention: `EA = (X_h - mu_h)^2 / (v_h + eps)` (high-frequency energy highlights boundaries).
- Spatial attention: `SA = sigmoid((X_l^2 - mu_l) / (v_l + eps))` (low-frequency energy highlights regions).
- Fused attention `A = sigmoid(EA + SA)`, output `A * X`.

There are no learnable parameters, so the checkpoint layout equals the baseline's. Sanity check: `bash launch.sh pfesacheck` (`tools/check_pfesa.py`). More: [`docs/README_UNIMISS3D.md`](docs/README_UNIMISS3D.md).

## 5. Repository layout

```
lib/models/TMamba2D.py, TMamba3D.py     original T-Mamba (2D also has forward_features() for SSL)
lib/models/pfesa.py                     PFESA module                                    [3D novelty]
lib/models/TMambaPFESA3D.py             T-Mamba3D + PFESA on the 3 skips                [3D novelty]
lib/ssl/unimiss2d.py, dataset2d.py      UniMiSS-2D student/teacher pre-training         [2D novelty]
lib/ssl/unimiss3d.py, dataset.py        UniMiSS 3D variant (experiment)
pretrain_unimiss2d.py                   Stage 1 pre-training for 2D                     [2D novelty]
train.py, test.py, inference.py         train / evaluate / infer (2D and 3D)
Tim/                                    causal-conv1d 1.0.0 and mamba_ssm 1.0.1 sources (built locally)
scripts/                                one-command steps: setup, data, smoke, pretrain, finetune, test, pfesa
launch2d.sh, launch.sh                  2D launcher, 3D launcher
demo_infer_2d.py, demo_web.py           image in -> mask out (command line, small upload page)
demo/                                   Kaggle notebook (2D + 3D demo) and run3d.py
tools/                                  download_data.py, infer_visualize.py, check_pfesa.py, FLOPs, metrics
datasets/                               dataset folders (filled by tools/download_data.py)
pretrain/                               checkpoints (best_TMamba2D.pth included)
docs/                                   method notes, original README, result plots
```

## 6. Before you start: what computer do you need?

The Mamba layers are **custom CUDA code**, so this project cannot run on a CPU, on a Mac, or on a laptop without an NVIDIA GPU.

| Your situation | What to do |
|---|---|
| Linux PC/server with an NVIDIA GPU | Follow [section 7](#7-setup-on-your-own-machine-step-by-step) |
| Windows PC with an NVIDIA GPU | Use WSL2 (Ubuntu) with the NVIDIA driver, then follow section 7 inside Ubuntu. We have not tested this route ourselves |
| Laptop **without** an NVIDIA GPU (most laptops) | Use Kaggle, [section 8](#8-setup-without-a-gpu-run-on-kaggle), or a college/cloud GPU machine |

**Minimum for the full pipeline**
- NVIDIA GPU. A T4 (16 GB) runs the demo and inference. Training the 3D model and pre-training are far faster on an A100-class GPU; we used an A100 80 GB for the 3D runs.
- CUDA 11.8 toolkit (`nvcc`), Python 3.9, PyTorch 2.0.1 (cu118), gcc/g++ 11.
- Free disk space for the datasets (the 3D archive alone is about 6 GB) plus extracted copies and run outputs: plan for tens of GB.
- Internet access for the downloads.

## 7. Setup on your own machine (step by step)

### Step 1: get the code
```bash
git clone https://github.com/Maleshkumarvelmurugan1982/A-Unified-T-Mamba-Framework-for-2D-and-3D-Dental-Segmentation
cd A-Unified-T-Mamba-Framework-for-2D-and-3D-Dental-Segmentation
```

### Step 2: check your GPU and compiler
```bash
nvidia-smi          # must show your GPU and driver
nvcc --version      # should say release 11.8 (install the CUDA 11.8 toolkit if missing)
gcc --version       # gcc/g++ 11 is the safe choice for the build
```

### Step 3: create the environment
```bash
conda create -n tmamba python=3.9 -y
conda activate tmamba
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install transformers==4.30.2 tokenizers==0.13.3 gdown
```
`numpy` must stay at **1.23.5** (already pinned in `requirements.txt`): the 3D code uses `np.int`, which newer numpy removed.

### Step 4: build the Mamba CUDA kernels (slow, 30 to 40 minutes)
```bash
export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST=7.5          # your GPU: 7.5 = T4, 8.0 = A100, 8.6 = RTX 30xx, 8.9 = RTX 40xx, 9.0 = H100
cd Tim/causal-conv1d && pip install . --no-build-isolation && cd ../..
cd Tim/mamba         && pip install . --no-build-isolation && cd ../..
```

### Step 5: verify the build
```bash
python -c "import torch, causal_conv1d_cuda, selective_scan_cuda; print('Mamba CUDA kernels OK')"
```
Import `torch` first, as shown, otherwise you get a `libc10.so` error.

> **One-command alternative** (fresh Linux VM, root user): `bash launch2d.sh setup` runs the same steps and installs everything under `/root/tmamba`. It assumes it may write to `/root`; on a normal user account use Steps 3 to 5 above.

### Step 6: download the data
See [section 9](#9-datasets-download-and-folder-setup).

### Step 7: run the check and the experiments
See sections [10](#10-check-that-everything-works), [11](#11-run-the-2d-experiments) and [12](#12-run-the-3d-experiments).

## 8. Setup without a GPU: run on Kaggle

Kaggle gives a free GPU. The notebook `demo/T-Mamba_2D_3D_Full_Demo_Kaggle.ipynb` builds the whole environment itself and runs both demos.

1. Create a **Kaggle dataset** containing: `T-Mamba-3D_PFESA_FINAL.zip` (a zip of this repository with the `Tim/` folder), `best_TMamba2D.pth`, `best_TMambaPFESA3D.pth`, and one or more X-ray images (`.jpg` / `.png`). `best_TMamba3D.pth` is optional (the notebook downloads it if missing).
2. Open the notebook on Kaggle. Settings: **Accelerator = GPU**, **Internet = ON**. Add your dataset as input.
3. Click **Run All**. The first run takes about 45 to 60 minutes (it compiles the Mamba kernels); every step is skipped if already done.
4. Results appear under `/kaggle/working/demo_out` (2D) and `/kaggle/working/viz3d` (3D), plus `all_outputs.zip` (download link at the end). Click **Save Version** so the outputs stay after the session.

Notes: the GPU queue can be long at busy hours. The notebook was assembled from cells that each ran successfully on their own; it has not yet been run top to bottom as one notebook, so run it once before relying on it.

## 9. Datasets: download and folder setup

```bash
pip install gdown
python tools/download_data.py --what 2d      # 2D X-ray dataset (2 parts)
python tools/download_data.py --what 3d      # 3D CBCT dataset
python tools/download_data.py --what ckpt3d  # 3D baseline checkpoint (optional)
python tools/download_data.py --what all     # everything
```

| Item | Google Drive file id(s) | Ends up in |
|---|---|---|
| 2D Tooth2D-X-Ray-6k (2 parts) | `1YTa8Z5iv_pO0G-0KeU-yTritkodg7TMK`, `1CUq-MuMIr7eHKBVZ4VTh-m72pcLMhgp2` | `datasets/Tooth2D-X-Ray-6k/` |
| 3D CBCT (NC-release-data-checked) | `1ivIhA222AoFxzTYWVAC3yUFSsg5Waqqu` | `datasets/NC-release-data-checked/` |
| 3D baseline checkpoint | `1WQeyZLiUeiNm8onicyeB5-TipOPZuaHu` | `pretrain/best_TMamba3D.pth` |

Link format: `https://drive.google.com/file/d/<id>/view`. The script downloads, extracts, moves the folders to the right place and prints the file counts. Re-running it skips what is done.

**If Google Drive shows a quota error:** download the file in your browser, save it into `datasets/_downloads/` as `2d_part1`, `2d_part2`, `3d_data` (or `3d_ckpt`), then run the same command again.

Expected layout:
```
datasets/
  Tooth2D-X-Ray-6k/          train/{images,annotations}   test/{images,annotations}    (5,000 / 1,225)
  NC-release-data-checked/   train/{images,labels}        valid/{images,labels}        (*.nii.gz)
```
The public TED dataset of the original authors is on [Hugging Face](https://huggingface.co/datasets/Bryceee/TED).

> The launch scripts default to `/root/tmamba/datasets`. To use the folders inside the repo, prefix the launcher, e.g.
> `DATA2D=$PWD/datasets/Tooth2D-X-Ray-6k DATA=$PWD/datasets/NC-release-data-checked bash launch2d.sh finetune`,
> or use the direct `python` commands below.

## 10. Check that everything works

```bash
bash launch2d.sh smoke        # 1-epoch, 8-image test; must print "SMOKE TEST 2D PASSED"
bash launch.sh pfesacheck     # PFESA sanity check on GPU (3D)
```
Quick inference check with the included 2D weights (any X-ray image):
```bash
python demo_infer_2d.py --weight pretrain/best_TMamba2D.pth --input my_xray.png --out demo_out
```

## 11. Run the 2D experiments

**With the launcher**
```bash
bash launch2d.sh baseline    # baseline (30 epochs) + test
bash launch2d.sh pretrain    # UniMiSS-2D Stage 1 (label-free)
bash launch2d.sh finetune    # Stage 2, initialised from Stage 1
bash launch2d.sh test        # Stage 3: DSC / IoU / ACC / JI
bash launch2d.sh testbase    # test your own baseline at pretrain/best_TMamba2D.pth
```
Long runs: `nohup bash launch2d.sh pretrain > pretrain2d.log 2>&1 &` then `tail -f pretrain2d.log`.
Settings (environment variables): `SSL2D_EPOCHS`, `SSL2D_BATCH`, `BATCH2D`, `FT2D_EPOCHS`, `GPU`. Out of memory: lower `SSL2D_BATCH` and `BATCH2D`, and use the same `BATCH2D` for baseline and novelty so the comparison stays fair.

**Direct commands**
```bash
python train.py --dataset Tooth2D-X-Ray-6k --dimension 2d --model TMamba2D --scaling_version SMALL \
  --epoch 30 --multi_gpu False --batch_size 2 --dataset_path ./datasets/Tooth2D-X-Ray-6k
python test.py  --dataset Tooth2D-X-Ray-6k --dimension 2d --model TMamba2D --scaling_version SMALL \
  --dataset_path ./datasets/Tooth2D-X-Ray-6k --pretrain_weight pretrain/best_TMamba2D.pth
```
We trained with batch size 2 on a single GPU; the config default is 4. Keep it identical for baseline and novelty.

## 12. Run the 3D experiments

**With the launcher**
```bash
bash launch.sh data          # CBCT dataset + baseline checkpoint
bash launch.sh baseline      # T-Mamba3D train + test
bash launch.sh pfesa         # T-Mamba3D + PFESA train + test (resumes automatically after an interruption)
bash launch.sh testbase      # test your saved best_TMamba3D.pth (no training)
```
**Direct commands** (20 epochs; use `--model TMamba3D` for the baseline)
```bash
python train.py --dataset 3D-CBCT-Tooth --dimension 3d --model TMambaPFESA3D --scaling_version TINY \
  --epoch 20 --multi_gpu False --batch_size 6 --dataset_path ./datasets/NC-release-data-checked
python test.py  --dataset 3D-CBCT-Tooth --dimension 3d --model TMambaPFESA3D --scaling_version TINY \
  --dataset_path ./datasets/NC-release-data-checked --pretrain_weight best_TMambaPFESA3D.pth
```
`--resume <runs/.../latest_<model>.state> --pretrain_weight <latest_<model>.pth>` continues an interrupted run. If the instance may be shut down (spot instances), re-run `bash launch.sh pfesa`; it picks up from the last finished epoch.

## 13. Live demo: image in, mask out

**2D, one image or a folder** (writes mask, overlay and a side-by-side picture)
```bash
python demo_infer_2d.py --weight pretrain/best_TMamba2D.pth --input my_xray.png --out demo_out
python demo_web.py        # small upload page on port 8000 (run on the GPU machine, open in a browser)
```
**3D, baseline vs PFESA on validation volumes** (slice overlays and 3D renders)
```bash
python tools/infer_visualize.py --data ./datasets/NC-release-data-checked \
  --base pretrain/best_TMamba3D.pth --pfesa best_TMambaPFESA3D.pth --out viz3d --n 2
```
In the 3D picture panels: green = correctly segmented tooth, red = false positive, blue = missed tooth.
No GPU on your side? Use the Kaggle notebook ([section 8](#8-setup-without-a-gpu-run-on-kaggle)).

## 14. Evaluation metrics

| | Metrics |
|---|---|
| 2D | DSC (Dice), IoU, ACC (pixel accuracy), JI |
| 3D | DSC, IoU, SO (overlap), HD (Hausdorff distance), ASSD (average symmetric surface distance) |

Higher is better for DSC / IoU / ACC / JI / SO; lower is better for HD and ASSD. HD and ASSD depend on voxel spacing and outlier handling, so compare them only within the same pipeline.
The original trainer evaluates the test split every epoch and keeps the best epoch as `best_*.pth`. For a paper, also report the last-epoch checkpoint of baseline and novelty, because best-epoch-on-test is optimistic.

## 15. Important implementation notes

- **Seed before model creation.** `SpectralGatingBlocks`, `GateModules` and the positional embeddings are stored in plain Python lists, so they are **not saved in the checkpoint**; they are re-created from the random seed. Always call `random / np / torch / cuda manual_seed(1777777)` immediately before building the model (as `test.py` does). Otherwise predictions will not match the trained model.
- **Preprocessing (2D):** resize to 640x1280, normalise with mean (0.50297405, 0.54711632, 0.71049083) and std (0.18653496, 0.17118206, 0.17080363).
- **Real CUDA kernels are required.** A pure-PyTorch replacement of Mamba is possible but was not used for any reported number.

## 16. Troubleshooting

| Problem | Fix |
|---|---|
| `AttributeError: module 'numpy' has no attribute 'int'` | `pip install numpy==1.23.5` |
| `ImportError: libc10.so` when importing the CUDA extensions | `import torch` first, then `causal_conv1d_cuda` / `selective_scan_cuda` |
| Triton: `cannot find -lcuda` | link `libcuda.so` into a folder and add it to `LIBRARY_PATH` and `LD_LIBRARY_PATH` |
| `causal_conv1d` / `mamba_ssm` build fails | use gcc/g++ 11 and the CUDA 11.8 `nvcc`, set `TORCH_CUDA_ARCH_LIST` to your GPU, read the full build log (do not only look at the last lines) |
| matplotlib "invalid MPLBACKEND" in a subprocess | run with `MPLBACKEND=Agg` |
| CUDA out of memory | lower `BATCH` / `BATCH2D` / `SSL2D_BATCH` |
| Predicted mask looks wrong | check the seed note in section 15 and the image size and normalisation |
| `git push` rejected (fetch first) | `git pull origin main --allow-unrelated-histories -X ours --no-edit`, then push |
| Google Drive quota error | download manually into `datasets/_downloads/` (section 9) |

## 17. Weights

All checkpoints are in [`pretrain/`](pretrain) (each about 4 to 5.4 MB):

| File | Model |
|---|---|
| `pretrain/best_TMamba2D.pth` | T-Mamba2D (2D demo and test) |
| `pretrain/best_TMamba3D.pth` | T-Mamba3D baseline (3D) |
| `pretrain/best_TMambaPFESA3D.pth` | T-Mamba3D + PFESA (3D) |

Load them only after setting the seed as described in [section 15](#15-important-implementation-notes). Larger files in future should go to the GitHub **Releases** page (GitHub blocks files above 100 MB).

## 18. Credits and licence

- **T-Mamba** (base code and model): original authors, MIT licence, see [`LICENSE`](LICENSE) and [`docs/README_original_TMamba.md`](docs/README_original_TMamba.md).
- **UniMiSS:** Xie et al., ECCV 2022 (idea behind UniMiSS-2D).
- **PFESA:** Li et al., MICCAI 2025 (idea behind our 3D module).
- Our additions are released under the same MIT licence.
