"""
UniMiSS (Xie et al.) -> 3D adaptation for T-Mamba.

Kept from UniMiSS:
  * student / teacher self-distillation (DINO style: centering + sharpening, EMA teacher, stop-grad)
  * volume-slice consistency: the 8-term objective of Eq.(6) between
        f  = volume representation          (global, whole 3D crop)
        f' = mean of "slice" representations (here: mean of K depth-slab representations)
    i.e. volume<->volume, slice<->slice and volume<->slice are all pulled together across two views.

3D-only adaptation (documented honestly):
  The T-Mamba encoder has fixed-size positional embeddings, so it cannot be re-run on separate 2D slices
  the way UniMiSS' dimension-free MiT can. The "slice branch" is therefore obtained from the SAME 3D forward
  pass by pooling the encoder feature maps over K non-overlapping slabs along the slice axis and passing each
  slab descriptor through the shared projector; f' is the average over slabs (same averaging as UniMiSS).
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=1024, bottleneck_dim=256, norm_last_layer=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim))
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class UniMiSS3D(nn.Module):
    """backbone (TMamba3D) + projector. forward(x) -> (volume_logits, slab_logits), both (B, out_dim)."""

    def __init__(self, backbone, out_dim=4096, num_slabs=4, slab_axis=-1):
        super().__init__()
        self.backbone = backbone
        self.num_slabs = num_slabs
        self.slab_axis = slab_axis
        feat_dim = int(sum(backbone.feature_channels))
        self.norm = nn.LayerNorm(feat_dim)
        self.head = DINOHead(feat_dim, out_dim)

    def forward(self, x):
        feats = self.backbone.forward_features(x)  # list of (B,C,a,b,c)
        B = x.shape[0]
        vol = torch.cat([f.mean(dim=(2, 3, 4)) for f in feats], dim=1)  # (B, sumC)
        # slice branch: per-slab descriptors
        per_scale = []
        for f in feats:
            parts = torch.tensor_split(f, self.num_slabs, dim=self.slab_axis)
            per_scale.append(torch.stack([p.mean(dim=(2, 3, 4)) for p in parts], dim=1))  # (B,K,C)
        slab = torch.cat(per_scale, dim=2)  # (B,K,sumC)
        v_logits = self.head(self.norm(vol))
        s_logits = self.head(self.norm(slab.reshape(B * self.num_slabs, -1)))
        s_logits = s_logits.reshape(B, self.num_slabs, -1).mean(dim=1)  # average over slabs (UniMiSS f')
        return v_logits, s_logits


class UniMiSSLoss(nn.Module):
    """H(f_a, f_b) = - softmax((f_b - C)/tau_t) * log softmax(f_a/tau_s) ; 8-term symmetric objective (Eq. 6)."""

    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9, use_slab=True):
        super().__init__()
        self.student_temp, self.teacher_temp, self.m, self.use_slab = student_temp, teacher_temp, center_momentum, use_slab
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, s_out, t_out):
        """s_out / t_out: (vol_logits, slab_logits), each (2B, K) with the first B rows = view1, last B = view2."""
        kinds = (0, 1) if self.use_slab else (0,)
        S = [[o.chunk(2)[v] for v in range(2)] for o in s_out]   # S[kind][view]
        T = [[o.chunk(2)[v] for v in range(2)] for o in t_out]
        total, n = 0.0, 0
        for vs in range(2):          # student view
            for vt in range(2):      # teacher view
                if vs == vt:
                    continue
                for ks in kinds:
                    for kt in kinds:
                        t_prob = F.softmax((T[kt][vt] - self.center) / self.teacher_temp, dim=-1).detach()
                        s_logp = F.log_softmax(S[ks][vs] / self.student_temp, dim=-1)
                        total = total + torch.sum(-t_prob * s_logp, dim=-1).mean()
                        n += 1
        self._update_center(t_out)
        return total / n

    @torch.no_grad()
    def _update_center(self, t_out):
        batch_center = torch.cat(list(t_out), dim=0).mean(dim=0, keepdim=True)
        self.center = self.center * self.m + batch_center * (1 - self.m)


@torch.no_grad()
def ema_update(student, teacher, momentum):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_((1 - momentum) * ps.detach().data)


def cosine_schedule(base, final, total_iters, warmup_iters=0, warmup_start=0.0):
    warm = [warmup_start + (base - warmup_start) * i / max(warmup_iters, 1) for i in range(warmup_iters)]
    it = [final + 0.5 * (base - final) * (1 + math.cos(math.pi * i / max(total_iters - warmup_iters, 1)))
          for i in range(total_iters - warmup_iters)]
    return warm + it
