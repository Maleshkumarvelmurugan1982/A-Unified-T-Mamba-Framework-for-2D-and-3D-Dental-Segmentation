"""
UniMiSS (self-distillation) adapted to 2D X-rays for T-Mamba2D.
Kept from UniMiSS: student / EMA-teacher self-distillation (centering + sharpening) and an image<->part consistency
(the 2D counterpart of the paper's volume<->slice consistency): the GLOBAL image descriptor must agree with the average of
K REGION descriptors (strips along the width of the feature map), across two augmented views, with the 8-term loss.
NOT in 2D: the 2D/3D switchable embedding and X-ray+CT mixing (that needs a dimension-free encoder, T-Mamba is not).
"""
import torch
import torch.nn as nn
from .unimiss3d import DINOHead, UniMiSSLoss, ema_update, cosine_schedule  # same head / loss / EMA as the 3D version


class UniMiSS2D(nn.Module):
    def __init__(self, backbone, out_dim=4096, num_regions=4, region_axis=-1):
        super().__init__()
        self.backbone, self.k, self.axis = backbone, num_regions, region_axis
        d = int(sum(backbone.feature_channels))
        self.norm = nn.LayerNorm(d)
        self.head = DINOHead(d, out_dim)

    def forward(self, x):
        feats = self.backbone.forward_features(x)                  # list of (B,C,h,w)
        B = x.shape[0]
        glob = torch.cat([f.mean(dim=(2, 3)) for f in feats], 1)   # (B, sumC)
        per = []
        for f in feats:
            parts = torch.tensor_split(f, self.k, dim=self.axis)
            per.append(torch.stack([p.mean(dim=(2, 3)) for p in parts], 1))  # (B,K,C)
        reg = torch.cat(per, 2)                                    # (B,K,sumC)
        g = self.head(self.norm(glob))
        r = self.head(self.norm(reg.reshape(B * self.k, -1))).reshape(B, self.k, -1).mean(1)
        return g, r
