"""Image-folder source: random ``patchsize``-crops from a directory of images,
returned as float tensors in [0, 1]. Ported from ``image_data.py``; requires
Pillow and torchvision (imported lazily by ``rdsandwich.dataloader``).
"""
from __future__ import annotations

import glob
import os
import random
from typing import List, Optional

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from .base import Source


class ImageFolderSource(Source):
    def __init__(self, root_or_glob: str, patchsize: Optional[int] = 256, device=None):
        if os.path.isdir(root_or_glob):
            self.paths: List[str] = sorted(
                glob.glob(os.path.join(root_or_glob, "*.png"))
                + glob.glob(os.path.join(root_or_glob, "*.jpg"))
            )
        else:
            self.paths = sorted(glob.glob(root_or_glob))
        assert self.paths, f"No images found under {root_or_glob!r}"
        self.patchsize = patchsize
        self.device = device

    def _load(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        t = TF.to_tensor(img)  # [3, H, W] in [0, 1]
        if self.patchsize is not None:
            _, h, w = t.shape
            ps = self.patchsize
            if h < ps or w < ps:
                t = TF.resize(t, [max(ps, h), max(ps, w)])
                _, h, w = t.shape
            top = random.randint(0, h - ps)
            left = random.randint(0, w - ps)
            t = t[:, top: top + ps, left: left + ps]
        return t

    def sample(self, batchsize: int) -> torch.Tensor:
        paths = random.choices(self.paths, k=batchsize)
        batch = torch.stack([self._load(p) for p in paths], dim=0)
        return batch.to(self.device) if self.device else batch

    def all_images(self):
        """Yield whole (un-cropped) images, one at a time — for Kodak/Tecnick-style
        full-image evaluation rather than patch sampling."""
        for p in self.paths:
            img = Image.open(p).convert("RGB")
            yield TF.to_tensor(img).unsqueeze(0)
