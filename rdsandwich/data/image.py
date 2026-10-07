"""Image-folder source: random ``patchsize``-crops from a directory of images,
returned as float tensors in **[0, 255]**. Ported from the authors'
``utils.get_custom_dataset`` / ``process_image``: images are read, randomly
cropped to ``patchsize``, and cast to float **without** rescaling -- the
``/255`` happens inside the model (``AnalysisTransform`` / the ResNet-VAE's
bottom ``EncoderBlock``), and the neural-compression convention throughout the
paper is to work in the ``[0, 255]`` range so MSE is in ``[0,255]^2`` units.

Decoding is the expensive part. By default an image is re-decoded from disk on
every access, so over a multi-epoch run the same PNG/JPEG is decoded again each
epoch (pure overhead). Passing ``preload=True`` decodes every image **once** at
construction and keeps it resident as a compact ``uint8`` CHW tensor; each later
access is then just a random crop (a cheap slice), with **no disk I/O or decode
after startup**. Storage is ``uint8`` (1 byte/subpixel), so a preloaded set costs
about ``sum(3 * H * W)`` bytes of RAM -- fine for finite eval-scale sets, but
check the budget for large training folders (see the RAM note on the class).

Requires Pillow and numpy; torchvision is imported lazily only for the rare
"image smaller than the patch" resize path.
"""
from __future__ import annotations

import glob
import os
import random
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from rdsandwich.data.base import Source
import torchvision.transforms.functional as TF


def _read_uint8(path: str) -> torch.Tensor:
    """Decode one image to a ``uint8`` CHW tensor (RGB).

    Uses Pillow's decoded array directly (no ``torchvision.to_tensor``, which
    would divide by 255 only for us to multiply it straight back). The file
    descriptor is released as soon as the pixels are read.
    """
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)  # [H, W, 3], writable copy
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3, H, W] uint8


SMALL_IMAGE_MODES = ("resize", "reflect_pad")


def _reflect_pad_to(t: torch.Tensor, size: int) -> torch.Tensor:
    """Mirror-pad a CHW tensor so both spatial sides are at least ``size``.

    Same as torchvision's ``RandomCrop(size, pad_if_needed=True, padding_mode='reflect')``
    on a PIL image: each too-short side is padded on *both* ends by its shortfall
    (``size - side``) with ``numpy.pad(mode="reflect")``, which keeps reflecting back
    and forth for pads longer than the image, so any image size works.
    """
    _, h, w = t.shape
    pad_h = size - h if h < size else 0
    pad_w = size - w if w < size else 0
    if not (pad_h or pad_w):
        return t
    arr = np.pad(t.numpy(), ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    return torch.from_numpy(arr)


def _random_crop(t: torch.Tensor, patchsize: Optional[int],
                 small_image_mode: str = "resize") -> torch.Tensor:
    """Take a random ``patchsize`` spatial crop from a CHW tensor.

    Dtype-agnostic (works on the stored ``uint8`` or on a float tensor). If the
    image is smaller than the patch it is first resized up (``small_image_mode=
    "resize"``, the original behaviour) or mirror-padded (``"reflect_pad"``, as in
    Duan et al.'s training; see ``_reflect_pad_to``). Returns a view (a plain
    slice) in the common case.
    """
    if patchsize is None:
        return t
    _, h, w = t.shape
    ps = patchsize
    if (h < ps or w < ps) and small_image_mode == "reflect_pad":
        t = _reflect_pad_to(t, ps)
        _, h, w = t.shape
    elif h < ps or w < ps:
        # Lazy import so the common crop-only path never needs torchvision.
        t = TF.resize(t, [max(ps, h), max(ps, w)], antialias=True)
        _, h, w = t.shape
    top = random.randint(0, h - ps)
    left = random.randint(0, w - ps)
    return t[:, top: top + ps, left: left + ps]


def _find_paths(root_or_glob: str) -> List[str]:
    if os.path.isdir(root_or_glob):
        paths = sorted(
            glob.glob(os.path.join(root_or_glob, "*.png"))
            + glob.glob(os.path.join(root_or_glob, "*.jpg"))
            + glob.glob(os.path.join(root_or_glob, "*.jpeg"))
        )
    else:
        paths = sorted(glob.glob(root_or_glob))
    if not paths:
        raise RuntimeError(f"No images found under {root_or_glob!r}")
    return paths


class ImageFolderDataset(Source, Dataset):
    """Images from a folder (or glob), as float tensors in [0, 255].

    One class serves every use of an image set:

      * map-style ``Dataset``: ``__len__`` is the number of images and
        ``__getitem__`` returns one random ``patchsize``-crop of that image, so a
        shuffling ``DataLoader`` passes over **every image once per epoch**
        (different epochs still see different crops);
      * ``sample(batchsize)``: random crops of images drawn with replacement
        (what the lower-bound evaluator draws its batches from);
      * ``all_images()``: whole, un-cropped images one at a time, for full-image
        (Kodak / Tecnick) evaluation.

    ``patchsize=None`` disables cropping. ``max_images`` optionally caps the set
    (e.g. for smoke tests). Training patches (``__getitem__`` / ``sample``) can be
    mirrored left-right with probability 0.5 (``hflip``), and images smaller than
    the patch are resized up or mirror-padded (``small_image_mode``, see
    ``_random_crop``). ``all_images()`` is never augmented.

    ``preload=True`` decodes the whole (capped) set into RAM **once** at
    construction, as ``uint8`` CHW tensors, so nothing is re-decoded afterwards
    -- an access becomes a random crop of an in-memory tensor. Pair it with
    ``num_workers=0``: there is no decode left to parallelise, and a single
    resident copy avoids per-worker duplication. RAM cost is about
    ``sum(3 * H * W)`` bytes over the kept images.
    """

    def __init__(self, root_or_glob: str, patchsize: Optional[int] = 256,
                 max_images: Optional[int] = None, preload: bool = False,
                 hflip: bool = False, small_image_mode: str = "resize"):
        if patchsize is not None and patchsize <= 0:
            raise ValueError("patchsize must be positive or None")
        if small_image_mode not in SMALL_IMAGE_MODES:
            raise ValueError(f"small_image_mode must be one of {SMALL_IMAGE_MODES}, "
                             f"got {small_image_mode!r}")
        self.hflip = hflip
        self.small_image_mode = small_image_mode
        paths = _find_paths(root_or_glob)
        self.paths = paths[:max_images] if max_images else paths
        self.patchsize = patchsize
        # Decode-once cache of uint8 CHW tensors, or None for lazy per-access decode.
        self._images: Optional[List[torch.Tensor]] = (
            [_read_uint8(p) for p in self.paths] if preload else None
        )

    def _image(self, index: int) -> torch.Tensor:
        """Full ``uint8`` CHW image for ``paths[index]`` (cached or freshly read)."""
        if self._images is not None:
            return self._images[index]
        return _read_uint8(self.paths[index])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        patch = _random_crop(self._image(index), self.patchsize, self.small_image_mode)
        if self.hflip and random.random() < 0.5:
            patch = patch.flip(-1)
        return patch.float()  # [3, ps, ps] in [0, 255]

    def sample(self, batchsize: int) -> torch.Tensor:
        if batchsize <= 0:
            raise ValueError("batchsize must be positive")
        idx = random.choices(range(len(self.paths)), k=batchsize)
        return torch.stack([self[i] for i in idx], dim=0)

    def all_images(self):
        """Yield whole (un-cropped) images as ``[1, 3, H, W]`` float tensors."""
        for i in range(len(self.paths)):
            yield self._image(i).float().unsqueeze(0)
