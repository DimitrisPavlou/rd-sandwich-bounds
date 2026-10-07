"""Upper-bound trainer: a thin ``BaseTrainer`` subclass for any beta-VAE.

The only thing asked of the model is ``get_losses(x) -> (loss, rate,
distortion)``; ``train_step`` returns that Lagrangian, and the rest of the
loop (batching, optimizer/scheduler step, logging, checkpointing) comes from
``BaseTrainer``. Optional speed-ups (``torch.compile``, channels_last) are
trainer options, so they apply to every model. The LR schedules live in
``rdsandwich.utils.lr_schedulers``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch._dynamo  # noqa: F401 (for torch._dynamo.config, used to raise the recompile limit)

from rdsandwich.utils.trainer import BaseTrainer


class UpperBoundTrainer(BaseTrainer):
    """Trains any model with ``get_losses(x) -> (loss, rate, distortion)`` to
    minimize rate + lambda * distortion.

    Extra options (all other keyword arguments go to ``BaseTrainer``):

    * ``compile``: wrap ``model.get_losses`` in ``torch.compile``. The trainer
      keeps the original module, so checkpoints and parameters stay clean.
    * ``channels_last``: use the channels_last memory format for the model and
      every batch (image models on tensor-core GPUs).
    * ``checkpoint_metadata``: extra entries stored in the checkpoint alongside
      the model config (e.g. ``{"model": "resnet_vae"}``).

    If the model defines ``diagnostics(x) -> dict``, the explosion report
    includes its output (e.g. per-level latent statistics).
    """

    def __init__(self, model, loader, *, compile: bool = False, channels_last: bool = False,
                 checkpoint_metadata: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(model, loader, **kwargs)
        self.channels_last = channels_last
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if compile:
            # Image models call shared sub-modules at many spatial resolutions in one
            # forward, so dynamo specializes one graph per resolution. That set is
            # finite but exceeds the default recompile_limit (8), after which dynamo
            # would fall back to eager; raise it so every resolution compiles once.
            torch._dynamo.config.recompile_limit = 128
            self._get_losses = torch.compile(self.model.get_losses)
        else:
            self._get_losses = self.model.get_losses

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(memory_format=torch.channels_last) if self.channels_last else x

    def train_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss, rate, distortion = self._get_losses(self._prepare(x))
        return loss, {"rate": rate.item(), "mse": distortion.item()}

    def diagnostic_forward(self, x: torch.Tensor) -> Dict[str, Any]:
        # Eager (uncompiled) model so the activation hooks fire.
        x = self._prepare(x)
        if hasattr(self.model, "diagnostics"):
            return self.model.diagnostics(x)
        self.model.get_losses(x)
        return {}

    def checkpoint_extra(self) -> Dict[str, Any]:
        cfg = getattr(self.model, "cfg", None)
        return {"cfg": vars(cfg) if cfg is not None and hasattr(cfg, "__dict__") else None,
                **self.checkpoint_metadata}
