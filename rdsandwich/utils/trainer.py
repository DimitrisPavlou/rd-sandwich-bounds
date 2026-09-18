"""A small, model-agnostic training loop shared by the bound trainers.

``BaseTrainer`` owns the generic mechanics — pulling batches from a
dataloader, the optimizer/scheduler step, per-epoch metric averaging, jsonl
logging, and checkpointing — so that a concrete trainer only has to say how a
single batch becomes a loss (``train_step``). The upper- and lower-bound
trainers subclass this; because the loop consumes any iterable of ``x``
batches, the same trainer works across every source in ``rdsandwich.dataloader``.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import IterableDataset

from .io import JsonlLogger, latest_checkpoint, load_checkpoint, save_checkpoint


class WarmupReduceLROnPlateau:
    """``ReduceLROnPlateau`` that ignores the first ``warmup`` epochs.

    Mirrors the original repo's ``MyReduceLROnPlateauCallback``: the learning
    rate is held constant during warmup, then reduced by ``factor`` after
    ``patience`` epochs without ``min_delta`` improvement, down to ``min_lr``.
    """

    def __init__(self, optimizer, *, factor=0.5, patience=10, warmup=100,
                 min_lr=1e-6, min_delta=1e-4, mode="min", verbose=False):
        self.optimizer = optimizer
        self.warmup = warmup
        self._plateau = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience,
            min_lr=min_lr, threshold=min_delta, threshold_mode="abs",
        )
        self.verbose = verbose
        self._epoch = 0

    def step(self, metric: float) -> None:
        if self._epoch >= self.warmup:
            self._plateau.step(metric)
        self._epoch += 1

    def state_dict(self):
        return {"epoch": self._epoch, "plateau": self._plateau.state_dict()}

    def load_state_dict(self, state):
        self._epoch = state.get("epoch", 0)
        if "plateau" in state:
            self._plateau.load_state_dict(state["plateau"])


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
        grad_clip: Optional[float] = None,
        amp: bool = False,
        checkpoint_interval: Optional[int] = None,
        val_fn: Optional[Callable[[], Dict[str, float]]] = None,
        monitor: str = "loss",
        terminate_on_nan: bool = True,
        resume: bool = False,
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
        self.grad_clip = grad_clip
        self.checkpoint_interval = checkpoint_interval
        self.val_fn = val_fn
        self.monitor = monitor
        self.terminate_on_nan = terminate_on_nan
        self.start_epoch = 0
        use_cuda = self.device.type == "cuda"
        self.amp = bool(amp and use_cuda)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        if resume:
            self._maybe_resume()
        # Only the infinite (analytic-source) path pulls batches through this
        # iterator; for a finite map-style DataLoader we iterate it directly each
        # epoch, so don't eagerly spawn its workers here.
        self._iter = iter(loader) if self._is_infinite_loader() else None

    # ----------------------------------------------------------------- #
    def _maybe_resume(self) -> None:
        """Load the newest checkpoint next to ``ckpt_path`` (model+optimizer+scheduler)."""
        if not self.ckpt_path:
            return
        ckpt_dir = os.path.dirname(self.ckpt_path) or "."
        path = latest_checkpoint(ckpt_dir)
        if path is None:
            return
        extra = load_checkpoint(path, self.model, self.optimizer, map_location=self.device)
        self.start_epoch = int(extra.get("epoch", 0)) + 1
        if self.scheduler is not None and extra.get("scheduler") is not None:
            try:
                self.scheduler.load_state_dict(extra["scheduler"])
            except Exception:  # noqa: BLE001 - resume best-effort
                pass
        if self.amp and extra.get("scaler") is not None:
            self.scaler.load_state_dict(extra["scaler"])
        if self.verbose:
            print(f"Resumed from {path} at epoch {self.start_epoch}")

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

    def _save(self, epoch: int, history: Dict[str, list], path: Optional[str] = None) -> None:
        path = path or self.ckpt_path
        if not path:
            return
        extra = {"history": dict(history), "epoch": epoch, **self.checkpoint_extra()}
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            extra["scheduler"] = self.scheduler.state_dict()
        if self.amp:
            extra["scaler"] = self.scaler.state_dict()
        save_checkpoint(path, self.model, self.optimizer, extra=extra)

    def _scheduler_step(self, monitor_value: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, (ReduceLROnPlateau, WarmupReduceLROnPlateau)):
            self.scheduler.step(monitor_value)
        else:
            self.scheduler.step()

    def train(self) -> Dict[str, list]:
        history: Dict[str, list] = defaultdict(list)
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            running: Dict[str, float] = defaultdict(float)
            n_steps = 0
            for x in self._epoch_batches():
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.amp):
                    loss, metrics = self.train_step(x)
                if self.terminate_on_nan and not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss ({loss.item()}) at epoch {epoch}, step {n_steps}."
                    )
                self.scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                running["loss"] += loss.item()
                for k, v in metrics.items():
                    running[k] += float(v)
                n_steps += 1
            for k in running:
                running[k] /= max(n_steps, 1)
                history[k].append(running[k])

            # Optional validation pass (drives plateau LR + checkpoint monitor).
            val_metrics: Dict[str, float] = {}
            if self.val_fn is not None:
                self.model.eval()
                val_metrics = {f"val_{k}": float(v) for k, v in self.val_fn().items()}
                for k, v in val_metrics.items():
                    history[k].append(v)

            monitor_value = val_metrics.get(f"val_{self.monitor}",
                                            running.get(self.monitor, running.get("loss", math.inf)))
            self._scheduler_step(monitor_value)

            if self.verbose:
                summary = " ".join(f"{k}={v:.5g}" for k, v in {**running, **val_metrics}.items())
                print(f"epoch {epoch}: {summary}")
            if self.logger:
                self.logger.log({"epoch": epoch, **running, **val_metrics,
                                 "lr": self.optimizer.param_groups[0]["lr"]})

            if self.checkpoint_interval and self.ckpt_path and (epoch + 1) % self.checkpoint_interval == 0:
                self._save(epoch, history)

        if self.logger:
            self.logger.close()
        self._save(self.epochs - 1, history)
        return dict(history)
