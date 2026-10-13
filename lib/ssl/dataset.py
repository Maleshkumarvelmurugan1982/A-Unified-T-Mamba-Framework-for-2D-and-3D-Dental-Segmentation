"""Unlabeled 3D CBCT dataset for UniMiSS-3D pre-training (labels are NOT used)."""
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from lib.utils.preprocess import load_image_or_label


def build_cache(image_paths, cache_dir, spacing, lo, hi, mean, std):
    """Resample + window + normalise every volume ONCE and store as float16 .npy (fast mmap reads later)."""
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for p in image_paths:
        name = os.path.basename(p).replace(".nii.gz", "").replace(".nii", "")
        cp = os.path.join(cache_dir, name + ".npy")
        if not os.path.isfile(cp):
            img, _ = load_image_or_label(p, spacing, type="image")
            img = (np.clip(img, lo, hi) - lo) / float(hi - lo)
            img = ((img - mean) / std).astype(np.float16)
            np.save(cp, img)
            print("[cache] {} -> {} {}".format(os.path.basename(p), cp, img.shape))
        out.append(cp)
    return out


class SSLVolumeDataset(Dataset):
    def __init__(self, cache_paths, crop_size=(160, 160, 96), samples_per_epoch=256,
                 n_candidates=8, view_shift=(24, 24, 16)):
        self.paths = cache_paths
        self.crop = tuple(crop_size)
        self.n = samples_per_epoch
        self.k = n_candidates
        self.shift = view_shift

    def __len__(self):
        return self.n

    def _get(self, vol, corner):
        sl = tuple(slice(c, c + s) for c, s in zip(corner, self.crop))
        return torch.from_numpy(np.ascontiguousarray(vol[sl]).astype(np.float32))

    def __getitem__(self, idx):
        rng = np.random
        vol = np.load(self.paths[rng.randint(len(self.paths))], mmap_mode="r")
        # pad (zeros = very low intensity after windowing) if a dim is smaller than the crop
        pads = [max(0, c - s) for c, s in zip(self.crop, vol.shape)]
        if any(pads):
            vol = np.pad(np.asarray(vol), [(0, p) for p in pads], mode="minimum")
        maxc = [s - c for s, c in zip(vol.shape, self.crop)]
        # choose the most "informative" (highest mean intensity) of k random crops -> avoids empty background
        best, best_score = None, -1e9
        for _ in range(self.k):
            corner = [rng.randint(0, m + 1) for m in maxc]
            sc = float(np.asarray(vol[corner[0]:corner[0] + self.crop[0]:4,
                                      corner[1]:corner[1] + self.crop[1]:4,
                                      corner[2]:corner[2] + self.crop[2]:4]).mean())
            if sc > best_score:
                best, best_score = corner, sc
        c1 = best
        c2 = [int(np.clip(c + rng.randint(-s, s + 1), 0, m)) for c, s, m in zip(c1, self.shift, maxc)]
        v1 = self._augment(self._get(vol, c1))
        v2 = self._augment(self._get(vol, c2))
        return v1.unsqueeze(0), v2.unsqueeze(0)

    # ---- light-weight 3D augmentations (tensor, (A,B,C)) ----
    def _augment(self, x):
        for ax in range(3):
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[ax])
        if np.random.rand() < 0.5:  # random-resized crop
            s = np.random.uniform(0.7, 1.0)
            size = [max(8, int(d * s)) for d in x.shape]
            st = [np.random.randint(0, d - z + 1) for d, z in zip(x.shape, size)]
            sub = x[st[0]:st[0] + size[0], st[1]:st[1] + size[1], st[2]:st[2] + size[2]]
            x = F.interpolate(sub[None, None], size=self.crop, mode="trilinear", align_corners=False)[0, 0]
        x = x * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.15, 0.15)  # contrast / brightness
        if np.random.rand() < 0.3:  # blur
            x = F.avg_pool3d(x[None, None], 3, 1, 1)[0, 0]
        if np.random.rand() < 0.5:  # gaussian noise
            x = x + torch.randn_like(x) * np.random.uniform(0.0, 0.1)
        return x
