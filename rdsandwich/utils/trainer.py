"""A small, model-agnostic training loop shared by the bound trainers.

``BaseTrainer`` owns the generic mechanics — pulling batches from a
dataloader, the optimizer/scheduler step, per-epoch metric averaging, jsonl
logging, and checkpointing — so that a concrete trainer only has to say how a
single batch becomes a loss (``train_step``). The upper- and lower-bound
trainers subclass this; because the loop consumes any iterable of ``x``
batches, the same trainer works across every source in ``rdsandwich.dataloader``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

from .io import JsonlLogger, save_checkpoint


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        loader,
        *,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        steps_per_epoch: int,
        device=None,
        scheduler=None,
        logger: Optional[JsonlLogger] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
    ):
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.loader = loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.logger = logger
        self.ckpt_path = ckpt_path
        self.verbose = verbose
        self._iter = iter(loader)

    # ----------------------------------------------------------------- #
    # Hooks for subclasses
    # ----------------------------------------------------------------- #
    def train_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the loss for one batch. Returns (loss, metrics_to_log)."""
        raise NotImplementedError

    def checkpoint_extra(self) -> Dict[str, Any]:
        """Extra payload to store alongside the final checkpoint (e.g. the config)."""
        return {}

    # ----------------------------------------------------------------- #
    def _next_batch(self) -> torch.Tensor:
        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            batch = next(self._iter)
        return batch.to(self.device)

    def _is_infinite_loader(self) -> bool:
        """True for analytic sources wrapped as an ``InfiniteBatchDataset``.

        Finite map-style datasets (physics/speech arrays, image folders) have a
        real length and can be exhausted; infinite ones cannot.
        """
        return isinstance(getattr(self.loader, "dataset", None), IterableDataset)

    def _epoch_batches(self):
        """Yield one epoch's batches (already on ``self.device``).

        * infinite source -> a fixed ``steps_per_epoch`` batches (re-wrapping the
          endless stream), since there is no dataset to "pass over";
        * finite source   -> a single full pass over the loader (a true epoch),
          so the model sees every example once per epoch.
        """
        if self._is_infinite_loader():
            for _ in range(self.steps_per_epoch):
                yield self._next_batch()
        else:
            for batch in self.loader:
                yield batch.to(self.device)

    def train(self) -> Dict[str, list]:
        history: Dict[str, list] = defaultdict(list)
        for epoch in range(self.epochs):
            self.model.train()
            running: Dict[str, float] = defaultdict(float)
            n_steps = 0
            for x in self._epoch_batches():
                self.optimizer.zero_grad()
                loss, metrics = self.train_step(x)
                loss.backward()
                self.optimizer.step()
                running["loss"] += loss.item()
                for k, v in metrics.items():
                    running[k] += float(v)
                n_steps += 1
            for k in running:
                running[k] /= max(n_steps, 1)
                history[k].append(running[k])
            if self.scheduler is not None:
                self.scheduler.step()
            if self.verbose:
                summary = " ".join(f"{k}={running[k]:.5g}" for k in running)
                print(f"epoch {epoch}: {summary}")
            if self.logger:
                self.logger.log({"epoch": epoch, **running,
                                 "lr": self.optimizer.param_groups[0]["lr"]})

        if self.logger:
            self.logger.close()
        if self.ckpt_path:
            save_checkpoint(self.ckpt_path, self.model, self.optimizer,
                            extra={"history": dict(history), **self.checkpoint_extra()})
        return dict(history)
