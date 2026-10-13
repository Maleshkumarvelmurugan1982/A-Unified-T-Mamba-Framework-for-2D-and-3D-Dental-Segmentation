# T-Mamba2D + UniMiSS-2D pretraining (2D X-ray segmentation)

Reference baseline (your reproduced T-Mamba2D SMALL, 640x1280): DSC 0.884233 | IoU 0.797599 | ACC 0.961985 | JI 0.796064 | 0.943 M params | 18.744 GFLOPs.
The architecture is NOT changed, so params / FLOPs stay exactly the same; only the initialisation changes.

## What is new (UniMiSS idea, 2D version)
Stage 1 `pretrain_unimiss2d.py`: self-supervised pretraining of the T-Mamba2D encoder, no labels, only `train/images`
(the test split is never used). Student / EMA-teacher self-distillation (centering + sharpening) over two strongly augmented views,
plus an image<->region consistency (global descriptor must agree with the mean of K=4 width-strip descriptors) = the 2D
counterpart of UniMiSS' volume<->slice consistency (8-term loss, Eq. 6).
Stage 2: fine-tune with the ORIGINAL recipe (30 epochs, AdamW lr 0.0075, MultiStepLR [24,28], DiceLoss, batch 4), initialised from the Stage-1 encoder.
Stage 3: test -> valid_DSC / valid_IoU / valid_ACC / valid_JI.
Not in 2D: UniMiSS' 2D/3D switchable embedding and X-ray+CT mixing (needs a dimension-free encoder; T-Mamba is not).

## Run (fresh GPU VM, terminal only)
```
git clone <your-repo-url> && cd T-Mamba-Unified
bash launch2d.sh setup        # env + mamba build (skip if the VM already has the tmamba env from the 3D package)
bash launch2d.sh data         # downloads Tooth2D-X-Ray-6k, prints train/test counts
bash launch2d.sh smoke        # 1-epoch check -> must print SMOKE TEST 2D PASSED
nohup bash launch2d.sh pretrain > pretrain2d.log 2>&1 &    # Stage 1 (30 epochs x 1000 images)
nohup bash launch2d.sh finetune > finetune2d.log 2>&1 &    # Stage 2 (after Stage 1 finished)
bash launch2d.sh test         # Stage 3
```
`bash launch2d.sh testbase` tests your own baseline: put it at `pretrain/best_TMamba2D.pth` first.
Settings in `scripts/common.sh` (override via env): `SSL2D_EPOCHS`, `SSL2D_BATCH` (pairs/step), `BATCH2D`, `FT2D_EPOCHS`, `GPU`.
Extra unlabeled X-rays (e.g. TED3 unlabelled images) help pretraining: `EXTRA_UNLABELED=/path/to/xrays bash launch2d.sh pretrain`.
Learning-rate option for fine-tuning: `LR=0.002 bash launch2d.sh finetune` (default keeps the original 0.0075 so the comparison with your baseline is identical).
Out of memory? Try `SSL2D_BATCH=1` (one image pair = 2 images per step, still fine for BatchNorm) and `BATCH2D=2` (use the same BATCH2D for baseline and novelty).

## Fair comparison / protocol note
The original T-Mamba trainer evaluates the test split every epoch and keeps `best_TMamba2D.pth` = best epoch on that split.
Your baseline number was most likely produced the same way, so comparing like with like is fair; for the paper ALSO report the last-epoch
checkpoint (`latest`/final) of both, because best-epoch-on-test is optimistic.
`--init_weight <baseline.pth>` / `FT_INIT=<baseline.pth>` start from your trained baseline instead (cheaper, but extra training could explain a gain).

## Caveats
Untested on a GPU. Gains from short pretraining may be small or zero. Run the smoke test first and read the per-epoch time + peak GPU memory it prints.
