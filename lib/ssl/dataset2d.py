"""Unlabeled 2D X-ray dataset for UniMiSS-2D pre-training (labels / annotations are NOT used)."""
import glob
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import lib.transforms.two as T


def list_images(*dirs):
    out = []
    for d in dirs:
        for ext in ("jpg", "jpeg", "png", "bmp"):
            out += glob.glob(os.path.join(d, "**", "*." + ext), recursive=True)
    return sorted(set(out))


class SSLImageDataset(Dataset):
    def __init__(self, paths, size=(640, 1280), means=(0.50297405, 0.54711632, 0.71049083),
                 stds=(0.18653496, 0.17118206, 0.17080363), samples_per_epoch=1000):
        self.paths, self.n = paths, samples_per_epoch
        # same building blocks / normalisation as the supervised pipeline, but stronger (DINO-style) augmentation
        self.tf = T.Compose([
            T.RandomResizedCrop(size, scale=(0.25, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
            T.RandomGaussianNoise(p=0.3),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.Cutout(p=0.3, value=(0, 0)),
            T.ToTensor(),
            T.Normalize(mean=means, std=stds),
        ])

    def __len__(self):
        return self.n

    def _load(self, p):
        im = cv2.imread(p, -1)
        if im.dtype == np.uint16:
            im = (im / 257.0).astype(np.uint8)
        if im.ndim == 2:
            im = np.stack([im] * 3, -1)
        return np.ascontiguousarray(im[:, :, :3])

    def __getitem__(self, i):
        im = self._load(self.paths[np.random.randint(len(self.paths))])
        dummy = np.zeros(im.shape[:2], dtype=np.uint8)   # transforms expect a label; unused
        v1, _ = self.tf(im.copy(), dummy.copy())
        v2, _ = self.tf(im.copy(), dummy.copy())
        return v1, v2
