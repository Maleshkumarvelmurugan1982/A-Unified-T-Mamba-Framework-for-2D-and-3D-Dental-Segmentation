# T-Mamba + UniMiSS-3D (volume-slice consistency self-distillation)

## What was added
| File | Purpose |
|---|---|
| `pretrain_unimiss3d.py` | Stage-1 SSL pre-training script |
| `lib/ssl/unimiss3d.py` | student/teacher (EMA), DINO head, 8-term volume<->slice loss (UniMiSS Eq. 6) |
| `lib/ssl/dataset.py` | unlabeled 3D crops, two augmented views, cached preprocessing |
| `lib/models/TMamba3D.py` | + `forward_features()` (encoder-only, 3 scales); forward() unchanged |
| `lib/utils/checkpoint.py` | loads full checkpoints OR bare state_dicts (used by train/test) |
| `train.py/test.py/inference.py` | new `--dataset_path`, `--gpu` (no more hard-coded paths) |
| `ToothDataset.py` | fixed tuple-unpack bug in `__getitem__` |

## Dataset layout expected
```
<DATA>/train/images/*.nii.gz   <DATA>/train/labels/*.nii.gz   (same sorted order / matching names)
<DATA>/valid/images/*.nii.gz   <DATA>/valid/labels/*.nii.gz
```
Labels: any value >0 is treated as foreground (binary, `classes=2`).

## Run (fresh A100 VM, terminal only)
```
git clone <your-repo-url> && cd T-Mamba-Unified
nohup bash launch.sh > launch.log 2>&1 &     # full pipeline
tail -f launch.log
```
`launch.sh` runs: setup (env + mamba build) -> data (gdown + checks) -> smoke test -> SSL pretrain -> fine-tune -> test.
Single steps: `bash launch.sh setup|data|smoke|pretrain|finetune|test|baseline` (or call `scripts/*.sh` directly).
`baseline` trains plain T-Mamba with identical settings and tests it, for the comparison table.
Settings (paths, GPU id) are in `scripts/common.sh`; override e.g. `GPU=1 bash launch.sh pretrain`.
Logs: `/root/tmamba/logs/`.

## Notes / honest caveats
* The slice branch is computed from depth-slab pooling of the same 3D forward pass (T-Mamba has fixed positional
  embeddings so it cannot be re-run on 2D slices like UniMiSS' MiT). 2D X-ray part of UniMiSS is not used (3D only).
* Ablations: `--no_slice_consistency` (plain DINO), `--init_weight <your.pth>`, `--num_slabs`, `--slab_axis`.
* Slab axis: default is the last (96-voxel) axis of the crop. `num_slabs` must divide 48/24/12 -> use 2,3,4,6,12.
* Memory: batch 2 means 4 volumes through student (with grad) + teacher; keep batch_size >= 2 (BatchNorm).


## GPU notes (RTX 4090 etc.)
* Out of memory? `BATCH=3 SSL_BATCH=2 bash launch.sh ...` (keep BATCH the same for the baseline and the UniMiSS run, so the comparison is fair).
  The pretraining log prints "peak GPU mem" every epoch; for fine-tuning watch `nvidia-smi`.
* RTX 40-series (sm_89) needs the Triton/ptxas fix; `scripts/common.sh` applies it automatically (`fix_triton_for_ada`).

## Pretraining length
Default is a single run: **30 pretraining epochs** (256 crops/epoch, batch 3) then 20 fine-tuning epochs.
Change with `SSL_EPOCHS=<N>`; the encoder goes to `/root/tmamba/runs_ssl/unimiss3d_ep<N>/`.
`bash scripts/run_test.sh <path/to/best_TMamba3D.pth>` tests a specific run (default = newest run).

## You already have a trained baseline (best_TMamba3D.pth)
* `bash launch.sh data` downloads it to `pretrain/best_TMamba3D.pth`. `bash launch.sh testbase` evaluates it -> your baseline table. No baseline retraining.
* **Clean mode (default, recommended for the paper):** `bash launch.sh` -> SSL pretrain from random init (30 ep) -> fine-tune 20 ep -> test. Compare with `testbase`.
* **Fast mode:** `INIT_BEST=1 FT_EPOCHS=10 bash launch.sh pretrain` then `INIT_BEST=1 FT_EPOCHS=10 bash launch.sh finetune` then `bash launch.sh test`.
  Cheaper, but the model gets extra training on top of the baseline, so a gain may come from that extra training and not only from the novelty. State this in the paper if you use it.
