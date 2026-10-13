"""
PFESA: FFT-based Parameter-Free Edge and Structure Attention (Li et al., MICCAI 2025), 3D version.
Implemented from the paper's Eq.(1)-(5); the paper only says "3D follows the same principle", so the choices marked
[ASSUMPTION] below are MY interpretation, not taken from the paper's code.

  X (B,C,D,H,W) --FFT--> Gaussian low-pass G (sigma = r * min(dims)/max(dims), r = 0.1) --> X_l = iFFT(G*F),  X_h = iFFT((1-G)*F)
  EA = (X_h - mu_h)^2 / var_h                       (Eq. 2)   edge attention, high frequencies
  SA = sigmoid((X_l^2 - mu_l) / var_l)              (Eq. 4)   structure attention, low frequencies
  A  = sigmoid(EA + SA);   X_out = A * X            (Eq. 5)
No trainable parameters, no state_dict entries (so a trained T-Mamba checkpoint loads into the PFESA model unchanged).

[ASSUMPTION] frequency coordinates are normalised to [-1, 1) per axis (so sigma=0.1 is 10% of the half-bandwidth);
[ASSUMPTION] mu / var are per-sample, per-channel statistics over the spatial dims (like SimAM);
[ASSUMPTION] a small eps is added to the variances for numerical safety; the FFT is always computed in float32.
Ablation switches (env vars, default = paper): PFESA_VARIANT = full | ea | sa ; PFESA_R = cutoff ratio r.
"""
import os
import torch
import torch.nn as nn


def gaussian_lowpass(shape, r, device):
    """Gaussian low-pass mask in the *unshifted* FFT layout (matches torch.fft.fftn output order)."""
    grids = torch.meshgrid(*[torch.fft.fftfreq(n, d=1.0, device=device) * 2.0 for n in shape], indexing="ij")  # each in [-1, 1)
    sigma = r * min(shape) / max(shape)
    d2 = sum(g * g for g in grids)
    return torch.exp(-d2 / (2.0 * sigma * sigma))


class PFESA(nn.Module):
    def __init__(self, r=None, variant=None, eps=1e-5):
        super().__init__()
        self.r = float(os.environ.get("PFESA_R", 0.1)) if r is None else r
        self.variant = os.environ.get("PFESA_VARIANT", "full") if variant is None else variant
        assert self.variant in ("full", "ea", "sa"), self.variant
        self.eps = eps
        self._masks = {}

    def _mask(self, shape, device):
        key = (tuple(shape), str(device))
        if key not in self._masks:
            self._masks[key] = gaussian_lowpass(shape, self.r, device)
        return self._masks[key]

    def decouple(self, x):
        """x float32 (B,C,*spatial) -> (x_low, x_high), x_low + x_high == x."""
        dims = tuple(range(2, x.ndim))
        g = self._mask(x.shape[2:], x.device)
        f = torch.fft.fftn(x, dim=dims)
        return torch.fft.ifftn(f * g, dim=dims).real, torch.fft.ifftn(f * (1.0 - g), dim=dims).real

    def attention(self, x):
        dims = tuple(range(2, x.ndim))
        xl, xh = self.decouple(x)
        mu_h = xh.mean(dim=dims, keepdim=True)
        var_h = xh.var(dim=dims, unbiased=False, keepdim=True)
        ea = (xh - mu_h) ** 2 / (var_h + self.eps)                        # Eq. (2)
        mu_l = xl.mean(dim=dims, keepdim=True)
        var_l = xl.var(dim=dims, unbiased=False, keepdim=True)
        sa = torch.sigmoid((xl ** 2 - mu_l) / (var_l + self.eps))         # Eq. (4)
        if self.variant == "ea":
            return torch.sigmoid(ea)
        if self.variant == "sa":
            return torch.sigmoid(sa)
        return torch.sigmoid(ea + sa)                                     # Eq. (5)

    def forward(self, x):
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):    # FFT in fp32 (cuFFT half needs power-of-2 sizes)
            xf = x.float()
            out = self.attention(xf) * xf
        return out.to(dtype)
