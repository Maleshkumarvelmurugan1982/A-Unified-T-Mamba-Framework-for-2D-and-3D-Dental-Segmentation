"""GPU sanity check for PFESA. Usage: python tools/check_pfesa.py [--gpu 0] [--ckpt pretrain/best_TMamba3D.pth]"""
import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lib.models.TMamba3D import TMamba3D
from lib.models.TMambaPFESA3D import TMambaPFESA3D
from lib.models.pfesa import PFESA
from lib.utils.checkpoint import torch_load_any, extract_state_dict

ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--ckpt", default="")
a = ap.parse_args(); dev = torch.device(f"cuda:{a.gpu}")

# 1) math checks on random features
p = PFESA().to(dev)
x = torch.randn(2, 12, 80, 80, 48, device=dev)
xl, xh = p.decouple(x)
print("low+high reconstructs x, max err:", (xl + xh - x).abs().max().item())
A = p.attention(x)
print("A range: [%.3f, %.3f] (must be within [0.5,1))  NaN: %s" % (A.min().item(), A.max().item(), torch.isnan(A).any().item()))

# 2) params / weights
kw = dict(in_channels=1, classes=2, input_size=(160, 160, 96), high_freq=0.9, low_freq=0.1)
import inspect
try:
    base = TMamba3D(**kw).to(dev); pf = TMambaPFESA3D(**kw).to(dev)
except TypeError:
    kw.pop("classes"); kw["classes"] = 1
    base = TMamba3D(**kw).to(dev); pf = TMambaPFESA3D(**kw).to(dev)
nb = sum(q.numel() for q in base.parameters()); npf = sum(q.numel() for q in pf.parameters())
print("params baseline %d | PFESA %d | equal: %s" % (nb, npf, nb == npf))
if a.ckpt and os.path.exists(a.ckpt):
    sd = extract_state_dict(torch_load_any(a.ckpt))
    r = pf.load_state_dict(sd, strict=False)
    print("baseline ckpt -> PFESA model: missing", len(r.missing_keys), "unexpected", len(r.unexpected_keys))

# 3) forward/backward, memory, time
for name, m in (("baseline", base), ("PFESA", pf)):
    m.train(); torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize()
    xin = torch.randn(1, 1, 160, 160, 96, device=dev)
    t = time.time()
    for _ in range(3):
        out = m(xin); out.float().mean().backward(); m.zero_grad()
    torch.cuda.synchronize()
    print("%-8s out %s | %.2fs/iter | peak %.2f GB" % (name, tuple(out.shape), (time.time() - t) / 3, torch.cuda.max_memory_allocated(dev) / 1e9))
print("PFESA CHECK DONE")
