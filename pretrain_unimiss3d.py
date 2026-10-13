"""
UniMiSS-3D self-supervised pre-training for T-Mamba (volume-slice consistency self-distillation).

Usage:
  python pretrain_unimiss3d.py --dataset_path /data/NC-release-data-checked --out_dir ./runs_ssl/unimiss3d --gpu 0
Then fine-tune:
  python train.py --dataset 3D-CBCT-Tooth --model TMamba3D --dimension 3d --dataset_path ... \
         --pretrain_weight ./runs_ssl/unimiss3d/unimiss_encoder_teacher.pth
"""
import os
import sys
import glob
import copy
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True, help="root with train/images/*.nii.gz (labels are not used)")
    p.add_argument("--use_valid_images", action="store_true", help="also use valid/images (unlabeled) - off by default to avoid any leakage")
    p.add_argument("--extra_unlabeled_dir", type=str, default=None, help="optional folder with extra *.nii.gz CBCT volumes")
    p.add_argument("--cache_dir", type=str, default=None, help="where preprocessed volumes are cached (default: <out_dir>/cache)")
    p.add_argument("--out_dir", type=str, default="./runs_ssl/unimiss3d")
    p.add_argument("--init_weight", type=str, default=None, help="optional T-Mamba checkpoint (.pth, full or bare) to start the backbone from")
    p.add_argument("--resume", type=str, default=None, help="resume SSL from <out_dir>/last.pth")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--samples_per_epoch", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--weight_decay_end", type=float, default=0.4)
    p.add_argument("--momentum_teacher", type=float, default=0.996)
    p.add_argument("--student_temp", type=float, default=0.1)
    p.add_argument("--teacher_temp", type=float, default=0.04)
    p.add_argument("--out_dim", type=int, default=4096)
    p.add_argument("--num_slabs", type=int, default=4, help="K slabs for the slice branch (must divide the slab-axis size at every scale: 96 -> 48/24/12 OK for 4)")
    p.add_argument("--slab_axis", type=int, default=-1, help="tensor axis (of B,C,A,B,C) treated as the slice axis; -1 = last (96-voxel) axis")
    p.add_argument("--no_slice_consistency", action="store_true", help="ablation: volume<->volume only (plain DINO)")
    p.add_argument("--clip_grad", type=float, default=3.0)
    p.add_argument("--freeze_last_layer_epochs", type=int, default=1)
    p.add_argument("--save_freq", type=int, default=5)
    p.add_argument("--seed", type=int, default=1777777)
    # preprocessing - identical to train.py
    p.add_argument("--crop_size", type=int, nargs=3, default=[160, 160, 96])
    p.add_argument("--resample_spacing", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    p.add_argument("--clip_lower", type=float, default=-1412)
    p.add_argument("--clip_upper", type=float, default=17943)
    p.add_argument("--norm_mean", type=float, default=0.05029342141696459)
    p.add_argument("--norm_std", type=float, default=0.028477091559295814)
    return p.parse_args()


def main():
    a = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    from lib.models.TMamba3D import TMamba3D
    from lib.ssl import UniMiSS3D, UniMiSSLoss, SSLVolumeDataset, build_cache, ema_update
    from lib.ssl.unimiss3d import cosine_schedule
    from lib.utils.checkpoint import load_pretrained_flexible, torch_load_any

    assert torch.cuda.is_available(), "TMamba3D (mamba_ssm) needs a CUDA GPU"
    dev = torch.device("cuda")
    os.makedirs(a.out_dir, exist_ok=True)

    # ---------------- data ----------------
    imgs = sorted(glob.glob(os.path.join(a.dataset_path, "train", "images", "*.nii.gz")))
    if a.use_valid_images:
        imgs += sorted(glob.glob(os.path.join(a.dataset_path, "valid", "images", "*.nii.gz")))
    if a.extra_unlabeled_dir:
        imgs += sorted(glob.glob(os.path.join(a.extra_unlabeled_dir, "*.nii.gz")))
    assert len(imgs) > 0, "no .nii.gz found under {}/train/images".format(a.dataset_path)
    print("[data] {} unlabeled volumes".format(len(imgs)))
    cache = build_cache(imgs, a.cache_dir or os.path.join(a.out_dir, "cache"), a.resample_spacing,
                        a.clip_lower, a.clip_upper, a.norm_mean, a.norm_std)
    ds = SSLVolumeDataset(cache, tuple(a.crop_size), a.samples_per_epoch)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                    pin_memory=True, drop_last=True, persistent_workers=a.num_workers > 0)

    # ---------------- model ----------------
    backbone = TMamba3D(in_channels=1, classes=2, input_size=tuple(a.crop_size), high_freq=0.9, low_freq=0.1)
    if a.init_weight:
        load_pretrained_flexible(backbone, a.init_weight)
    student = UniMiSS3D(backbone, a.out_dim, a.num_slabs, a.slab_axis).to(dev)
    teacher = copy.deepcopy(student).to(dev)
    for p_ in teacher.parameters():
        p_.requires_grad = False
    loss_fn = UniMiSSLoss(a.out_dim, a.student_temp, a.teacher_temp, use_slab=not a.no_slice_consistency).to(dev)

    reg = [p_ for n, p_ in student.named_parameters() if p_.requires_grad and p_.ndim > 1 and not n.endswith(".bias")]
    noreg = [p_ for n, p_ in student.named_parameters() if p_.requires_grad and not (p_.ndim > 1 and not n.endswith(".bias"))]
    opt = torch.optim.AdamW([{"params": reg}, {"params": noreg, "weight_decay": 0.0}])

    ipe = len(dl)
    total = a.epochs * ipe
    lr_s = cosine_schedule(a.lr, a.min_lr, total, a.warmup_epochs * ipe)
    wd_s = cosine_schedule(a.weight_decay, a.weight_decay_end, total)
    mom_s = cosine_schedule(a.momentum_teacher, 1.0, total)

    start_ep = 0
    if a.resume:
        ck = torch_load_any(a.resume)
        student.load_state_dict(ck["student"]); teacher.load_state_dict(ck["teacher"])
        opt.load_state_dict(ck["optimizer"]); loss_fn.load_state_dict(ck["loss"]); start_ep = ck["epoch"] + 1
        print("[resume] from epoch", start_ep)

    def export(tag=None):
        sd = {k: v.detach().cpu() for k, v in teacher.backbone.state_dict().items()
              if not k.startswith("out_conv")}   # segmentation head stays untouched for fine-tuning
        torch.save(sd, os.path.join(a.out_dir, "unimiss_encoder_teacher.pth"))
        if tag:
            torch.save(sd, os.path.join(a.out_dir, "unimiss_encoder_teacher_{}.pth".format(tag)))

    def save(ep):
        torch.save({"epoch": ep, "student": student.state_dict(), "teacher": teacher.state_dict(),
                    "optimizer": opt.state_dict(), "loss": loss_fn.state_dict(), "args": vars(a)},
                   os.path.join(a.out_dir, "last.pth"))

    # ---------------- train ----------------
    student.train(); teacher.train()
    for ep in range(start_ep, a.epochs):
        t0, run = time.time(), 0.0
        for it, (v1, v2) in enumerate(dl):
            g = ep * ipe + it
            for i, pg in enumerate(opt.param_groups):
                pg["lr"] = lr_s[g]
                if i == 0:
                    pg["weight_decay"] = wd_s[g]
            x = torch.cat([v1, v2], 0).to(dev, non_blocking=True)   # (2B,1,160,160,96)
            with torch.no_grad():
                t_out = teacher(x)
            s_out = student(x)
            loss = loss_fn(s_out, t_out)
            if not torch.isfinite(loss):
                print("non-finite loss, stopping"); sys.exit(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.clip_grad > 0:
                nn.utils.clip_grad_norm_(student.parameters(), a.clip_grad)
            if ep < a.freeze_last_layer_epochs:
                for n, p_ in student.named_parameters():
                    if "last_layer" in n:
                        p_.grad = None
            opt.step()
            ema_update(student, teacher, mom_s[g])
            run += loss.item()
            if it % 10 == 0:
                print("ep {:03d} it {:03d}/{} loss {:.4f} lr {:.2e}".format(ep, it, ipe, loss.item(), lr_s[g]), flush=True)
        print("== epoch {:03d} mean loss {:.4f} ({:.0f}s) | peak GPU mem {:.1f} GB".format(ep, run / ipe, time.time() - t0, torch.cuda.max_memory_allocated() / 1e9), flush=True)
        save(ep); export()
        if (ep + 1) % a.save_freq == 0:
            export("ep{:04d}".format(ep + 1))
    export("final")
    print("DONE. Fine-tune with --pretrain_weight {}".format(os.path.join(a.out_dir, "unimiss_encoder_teacher.pth")))


if __name__ == "__main__":
    main()
