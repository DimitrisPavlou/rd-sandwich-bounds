"""
GAN-generated-image source (Sec. 6.3), torch-ported from ``biggan.py``.

The original loads a pretrained BigGAN-deep-128 from TF-Hub. The direct
PyTorch equivalent is the ``pytorch-pretrained-biggan`` package (a faithful
port of the same DeepMind weights), so this module wraps that rather than
re-hosting model weights:

    pip install pytorch-pretrained-biggan

Controls the intrinsic dimension ``d`` exactly as in the paper/original
code: only the first ``d`` of the 128 noise dimensions are randomized per
sample, the rest are held at zero (Pope et al., 2021's method for
controlling a GAN's intrinsic image dimension).
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from .dataloader import Source


class BigGANSource(Source):
    def __init__(self, class_name: str, intrinsic_dim: int, truncation: float = 0.8, device=None):
        try:
            from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, one_hot_from_names
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "BigGANSource requires `pip install pytorch-pretrained-biggan`."
            ) from e
        self._truncated_noise_sample = truncated_noise_sample
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BigGAN.from_pretrained("biggan-deep-128").to(self.device).eval()
        self.class_vector = torch.from_numpy(one_hot_from_names([class_name], batch_size=1)).to(self.device)
        self.intrinsic_dim = intrinsic_dim
        self.truncation = truncation

    @torch.no_grad()
    def sample(self, batchsize: int) -> torch.Tensor:
        noise = self._truncated_noise_sample(truncation=self.truncation, batch_size=batchsize)
        noise = torch.from_numpy(noise).to(self.device)
        if self.intrinsic_dim < noise.shape[1]:
            noise[:, self.intrinsic_dim:] = 0.0
        class_vec = self.class_vector.repeat(batchsize, 1)
        img = self.model(noise, class_vec, self.truncation)  # in [-1, 1]
        return (img + 1.0) * 0.5  # map to [0, 1], matching the original's post_process_fun
