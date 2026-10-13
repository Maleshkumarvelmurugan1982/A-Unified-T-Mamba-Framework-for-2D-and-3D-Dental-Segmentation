"""
UniMiSS-2D self-supervised pre-training for T-Mamba2D (self-distillation + image<->region consistency) on X-rays.
Labels are NOT used. Only <dataset_path>/train/images (+ optional extra unlabeled folder); the test split is never touched.

  python pretrain_unimiss2d.py --dataset_path /root/tmamba/datasets/Tooth2D-X-Ray-6k --out_dir /root/tmamba/runs_ssl/unimiss2d_ep30
Then fine-tune with the ORIGINAL 2D recipe, e.g.
  python train.py --dataset Tooth2D-X-Ray-6k --dimension 2d --model TMamba2D --scaling_version SMALL --epoch 30 \
      --multi_gpu False --dataset_path <...> --pretrain_weight <out_dir>/unimiss_encoder_teacher.pth
"""
import os, sys, copy, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True, help="root with train/images/*.jpg (annotations are not used)")
    p.add_argument("--extra_unlabeled_dir", type=str, default=None, help="optional folder with extra unlabeled X-rays (e.g. TED3 unlabelled images)")
    p.add_argument("--out_dir", type=str, default="./runs_ssl/unimiss2d")
    p.add_argument("--scaling_version", type=str, default="SMALL")
    p.add_argument("--init_weight", type=str, default=None, help="optional T-Mamba2D checkpoint to start the backbone from")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--samples_per_epoch", type=int, default=1000, help="random images per epoch")
    p.add_argument("--batch_size", type=int, default=2, help="image PAIRS per step (>=2 for BatchNorm); the network sees 2x this many images")
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
    p.add_argument("--num_regions", type=int, default=4, help="K width-strips for the region branch (must divide 160, the narrowest feature width)")
    p.add_argument("--region_axis", type=int, default=-1)
    p.add_argument("--no_region_consistency", action="store_true", help="ablation: global<->global only (plain DINO)")
    p.add_argument("--clip_grad", type=float, default=3.0)
    p.add_argument("--freeze_last_layer_epochs", type=int, default=1)
    p.add_argument("--save_freq", type=int, default=5)
    p.add_argument("--seed", type=int, default=1777777)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--width", type=int, default=1280)
    return p.parse_args()


def main():
    a = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    from lib.models.TMamba2D import TMamba2D
    from lib.ssl.unimiss2d import UniMiSS2D, UniMiSSLoss, ema_update, cosine_schedule
    from lib.ssl.dataset2d import SSLImageDataset, list_images
    from lib.utils.checkpoint import load_pretrained_flexible, torch_load_any

    assert torch.cuda.is_available(), "TMamba2D (mamba_ssm) needs a CUDA GPU"
    dev = torch.device("cuda")
    os.makedirs(a.out_dir, exist_ok=True)

    dirs = [os.path.join(a.dataset_path, "train", "images")]
    if a.extra_unlabeled_dir:
        dirs.append(a.extra_unlabeled_dir)
    paths = list_images(*dirs)
    assert len(paths) > 0, "no images found in {}".format(dirs)
    print("[data] {} unlabeled images (test split NOT used)".format(len(paths)))
    ds = SSLImageDataset(paths, (a.height, a.width), samples_per_epoch=a.samples_per_epoch)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=True,
                    drop_last=True, persistent_workers=a.num_workers > 0)

    backbone = TMamba2D(in_channels=3, classes=2, scaling_version=a.scaling_version,
                        input_size=(a.height, a.width), high_freq=0.9, low_freq=0.1)
    if a.init_weight:
        load_pretrained_flexible(backbone, a.init_weight)
    student = UniMiSS2D(backbone, a.out_dim, a.num_regions, a.region_axis).to(dev)
    teacher = copy.deepcopy(student).to(dev)
    for p_ in teacher.parameters():
        p_.requires_grad = False
    loss_fn = UniMiSSLoss(a.out_dim, a.student_temp, a.teacher_temp, use_slab=not a.no_region_consistency).to(dev)

    is_reg = lambda n, p_: p_.ndim > 1 and not n.endswith(".bias")
    reg = [p_ for n, p_ in student.named_parameters() if p_.requires_grad and is_reg(n, p_)]
    noreg = [p_ for n, p_ in student.named_parameters() if p_.requires_grad and not is_reg(n, p_)]
    opt = torch.optim.AdamW([{"params": reg}, {"params": noreg, "weight_decay": 0.0}])

    ipe = len(dl); total = a.epochs * ipe
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
        sd = {k: v.detach().cpu() for k, v in teacher.backbone.state_dict().items() if not k.startswith("out_conv")}
        torch.save(sd, os.path.join(a.out_dir, "unimiss_encoder_teacher.pth"))
        if tag:
            torch.save(sd, os.path.join(a.out_dir, "unimiss_encoder_teacher_{}.pth".format(tag)))

    def save(ep):
        torch.save({"epoch": ep, "student": student.state_dict(), "teacher": teacher.state_dict(),
                    "optimizer": opt.state_dict(), "loss": loss_fn.state_dict(), "args": vars(a)},
                   os.path.join(a.out_dir, "last.pth"))

    student.train(); teacher.train()
    for ep in range(start_ep, a.epochs):
        t0, run = time.time(), 0.0
        for it, (v1, v2) in enumerate(dl):
            g = ep * ipe + it
            for i, pg in enumerate(opt.param_groups):
                pg["lr"] = lr_s[g]
                if i == 0:
                    pg["weight_decay"] = wd_s[g]
            x = torch.cat([v1, v2], 0).to(dev, non_blocking=True)   # (2B,3,640,1280)
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
            if it % 20 == 0:
                print("ep {:03d} it {:04d}/{} loss {:.4f} lr {:.2e}".format(ep, it, ipe, loss.item(), lr_s[g]), flush=True)
        print("== epoch {:03d} mean loss {:.4f} ({:.0f}s) | peak GPU mem {:.1f} GB".format(
            ep, run / ipe, time.time() - t0, torch.cuda.max_memory_allocated() / 1e9), flush=True)
        save(ep); export()
        if (ep + 1) % a.save_freq == 0:
            export("ep{:04d}".format(ep + 1))
    export("final")
    print("DONE. Fine-tune with --pretrain_weight {}".format(os.path.join(a.out_dir, "unimiss_encoder_teacher.pth")))


if __name__ == "__main__":
    main()
