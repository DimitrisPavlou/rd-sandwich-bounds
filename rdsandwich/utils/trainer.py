"""A small, model-agnostic training loop shared by the bound trainers.

``BaseTrainer`` owns the generic mechanics — pulling batches from a
dataloader, the optimizer/scheduler step, per-epoch metric averaging, jsonl
logging, and checkpointing — so that a concrete trainer only has to say how a
single batch becomes a loss (``train_step``). The upper- and lower-bound
trainers subclass this; because the loop consumes any iterable of ``x``
batches, the same trainer works across every source in ``rdsandwich.dataloader``.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import IterableDataset

from .io import JsonlLogger, MyJSONEncoder, get_time_str, latest_checkpoint, load_checkpoint, save_checkpoint


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
        max_nonfinite_skips: int = 0,
        amp_nonfinite_patience: int = 20,
        history_len: int = 50,
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
        # Non-finite gradients: without AMP, skip up to `max_nonfinite_skips` consecutive
        # such steps (0 = abort on the first). With AMP, the GradScaler overflowing and
        # skipping a step is normal; only `amp_nonfinite_patience` consecutive ones
        # (i.e. the scale has already been cut ~2^patience) count as an explosion.
        self.max_nonfinite_skips = max_nonfinite_skips
        self.amp_nonfinite_patience = amp_nonfinite_patience
        self._consecutive_nonfinite = 0
        # Last few steps (loss/metrics/grad norm) for the explosion report.
        self._recent_steps: deque = deque(maxlen=history_len)
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

    def diagnostic_forward(self, batch: torch.Tensor) -> Dict[str, Any]:
        """Eager forward pass for the explosion report (run in fp32 under no_grad on
        the offending batch, with activation hooks attached). Override to also
        return model-specific statistics; the return value must be JSON-serializable."""
        if hasattr(self.model, "get_losses"):
            self.model.get_losses(batch)
        else:
            self.model(batch)
        return {}

    def _grad_norm_and_clip(self, max_norm: float) -> float:
        """Total grad 2-norm; scale grads down to ``max_norm`` if it is exceeded.

        Unlike ``clip_grad_norm_``, leaves non-finite gradients untouched (it would
        multiply them by 0/NaN), so the explosion report sees the real values.
        """
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))).item()
        if math.isfinite(norm) and norm > max_norm:
            torch._foreach_mul_(grads, max_norm / (norm + 1e-6))
        return norm

    # ----------------------------------------------------------------- #
    # Explosion diagnostics
    # ----------------------------------------------------------------- #
    @staticmethod
    def _tensor_summary(t: torch.Tensor) -> Dict[str, Any]:
        t = t.detach().float()
        finite = torch.isfinite(t)
        n_bad = int((~finite).sum().item())
        out: Dict[str, Any] = {"numel": t.numel(), "nonfinite": n_bad,
                               "nan": int(torch.isnan(t).sum().item())}
        if n_bad < t.numel():
            tf = t[finite]
            out.update(norm=tf.norm().item(), absmax=tf.abs().max().item(),
                       min=tf.min().item(), max=tf.max().item())
        return out

    def _activation_scan(self, batch: torch.Tensor) -> Dict[str, Any]:
        """Re-run the model eagerly with forward hooks; report every module whose
        output is non-finite (in execution order) and the largest activations."""
        records = []

        def hook(name):
            def fn(_mod, _inp, out):
                if isinstance(out, torch.Tensor) and out.is_floating_point():
                    s = self._tensor_summary(out)
                    records.append((name, s))
            return fn

        handles = [m.register_forward_hook(hook(n)) for n, m in self.model.named_modules() if n]
        error = None
        extra: Dict[str, Any] = {}
        try:
            with torch.no_grad():
                extra = self.diagnostic_forward(batch)
        except Exception as e:  # noqa: BLE001 - diagnostics must never mask the real error
            error = repr(e)
        finally:
            for h in handles:
                h.remove()
        nonfinite = [{"module": n, **s} for n, s in records if s["nonfinite"]]
        largest = sorted((r for r in records if "absmax" in r[1]),
                         key=lambda r: -r[1]["absmax"])[:25]
        return {
            "first_nonfinite_module": nonfinite[0]["module"] if nonfinite else None,
            "nonfinite_modules": nonfinite[:50],
            "largest_activations": [{"module": n, **s} for n, s in largest],
            "model_stats": extra,
            "error": error,
        }

    def _explosion_report(self, kind: str, batch: torch.Tensor, epoch: int, step: int,
                          loss: torch.Tensor, grad_norm: Optional[float]) -> Optional[str]:
        """Dump everything useful about a non-finite loss/gradient to
        ``explosion-<time>.json`` (+ ``.pt`` with the batch and weights) next to
        the checkpoint. Returns the json path."""
        params = list(self.model.named_parameters())
        param_rows, grad_rows = [], []
        for name, p in params:
            ps = self._tensor_summary(p)
            param_rows.append({"param": name, **ps})
            if p.grad is not None:
                grad_rows.append({"param": name, **self._tensor_summary(p.grad)})
        bad_params = [r for r in param_rows if r["nonfinite"]]
        bad_grads = [r for r in grad_rows if r["nonfinite"]]
        top_grads = sorted((r for r in grad_rows if "norm" in r), key=lambda r: -r["norm"])[:25]
        top_params = sorted((r for r in param_rows if "absmax" in r), key=lambda r: -r["absmax"])[:25]

        report: Dict[str, Any] = {
            "kind": kind,
            "epoch": epoch,
            "step": step,
            "lr": self.optimizer.param_groups[0]["lr"],
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "grad_clip": self.grad_clip,
            "amp": self.amp,
            "amp_scale": self.scaler.get_scale() if self.amp else None,
            "consecutive_nonfinite": self._consecutive_nonfinite,
            "batch": self._tensor_summary(batch),
            "num_nonfinite_params": len(bad_params),
            "nonfinite_params": bad_params,
            "num_nonfinite_grads": len(bad_grads),
            "nonfinite_grads": bad_grads,
            "largest_grads": top_grads,
            "largest_params": top_params,
            "recent_steps": list(self._recent_steps),
        }
        report["forward_scan"] = self._activation_scan(batch)

        # Print a short human-readable summary; the json has the full detail.
        print(f"\n!!! {kind} at epoch {epoch}, step {step}: loss={report['loss']}, "
              f"grad_norm={grad_norm}, lr={report['lr']:.3g}")
        print(f"    non-finite params: {len(bad_params)}, non-finite grads: {len(bad_grads)}")
        for r in bad_grads[:10]:
            print(f"      grad  {r['param']}: {r['nonfinite']}/{r['numel']} non-finite")
        for r in top_grads[:5]:
            print(f"      |grad| {r['param']}: norm={r['norm']:.4g}")
        scan = report["forward_scan"]
        print(f"    first non-finite activation: {scan['first_nonfinite_module']}")

        if not self.ckpt_path:
            return None
        stem = os.path.join(os.path.dirname(self.ckpt_path) or ".",
                            f"explosion-e{epoch}-s{step}-{get_time_str()}")
        with open(stem + ".json", "w") as f:
            json.dump(report, f, indent=1, cls=MyJSONEncoder, default=str)
        # The offending batch + (pre-update) weights, to reproduce offline.
        torch.save({"batch": batch.detach().cpu(), "model_state_dict": self.model.state_dict(),
                    "epoch": epoch, "step": step}, stem + ".pt")
        print(f"    explosion report -> {stem}.json (+ .pt with batch and weights)")
        return stem + ".json"

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
            epoch_start = time.perf_counter()
            self.model.train()
            running: Dict[str, float] = defaultdict(float)
            n_steps = 0
            gn_sum, gn_max, n_gn, n_clipped, n_nonfinite = 0.0, 0.0, 0, 0, 0
            max_norm = self.grad_clip if self.grad_clip is not None else math.inf
            for x in self._epoch_batches():
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.amp):
                    loss, metrics = self.train_step(x)
                if self.terminate_on_nan and not torch.isfinite(loss):
                    self._explosion_report("non-finite loss", x, epoch, n_steps, loss, None)
                    raise FloatingPointError(
                        f"Non-finite loss ({loss.item()}) at epoch {epoch}, step {n_steps}."
                    )
                self.scaler.scale(loss).backward()
                # Unscale (no-op without AMP) so the norm/clip act on the true gradients.
                self.scaler.unscale_(self.optimizer)
                grad_norm = self._grad_norm_and_clip(max_norm)

                if math.isfinite(grad_norm):
                    self._consecutive_nonfinite = 0
                    gn_sum += grad_norm
                    gn_max = max(gn_max, grad_norm)
                    n_gn += 1
                    n_clipped += int(grad_norm > max_norm)
                elif self.terminate_on_nan:
                    self._consecutive_nonfinite += 1
                    n_nonfinite += 1
                    if self.amp:
                        # Normal GradScaler overflow: scaler.step() below skips the
                        # update and lowers the scale. Only persistent overflow is an issue.
                        if self._consecutive_nonfinite > self.amp_nonfinite_patience:
                            self._explosion_report("persistent non-finite gradients (amp)",
                                                   x, epoch, n_steps, loss, grad_norm)
                            raise FloatingPointError(
                                f"{self._consecutive_nonfinite} consecutive non-finite gradient "
                                f"steps under AMP at epoch {epoch}, step {n_steps}."
                            )
                    else:
                        self._explosion_report("non-finite gradients", x, epoch, n_steps,
                                               loss, grad_norm)
                        if self._consecutive_nonfinite > self.max_nonfinite_skips:
                            raise FloatingPointError(
                                f"Non-finite gradient norm ({grad_norm}) with finite loss "
                                f"({loss.item()}) at epoch {epoch}, step {n_steps}."
                            )
                        # Skip this update entirely so NaN/inf never reach the weights/Adam state.
                        self.optimizer.zero_grad(set_to_none=True)
                        n_steps += 1
                        continue
                self.scaler.step(self.optimizer)
                self.scaler.update()
                loss_val = loss.item()
                running["loss"] += loss_val
                for k, v in metrics.items():
                    running[k] += float(v)
                self._recent_steps.append({"epoch": epoch, "step": n_steps, "loss": loss_val,
                                           "grad_norm": grad_norm,
                                           **{k: float(v) for k, v in metrics.items()}})
                n_steps += 1
            n_updates = max(n_steps - (0 if self.amp else n_nonfinite), 1)
            for k in running:
                running[k] /= n_updates
                history[k].append(running[k])
            grad_stats = {
                "grad_norm_mean": gn_sum / max(n_gn, 1),
                "grad_norm_max": gn_max,
                "clip_frac": n_clipped / max(n_steps, 1),
                "nonfinite_grad_steps": n_nonfinite,
            }
            for k, v in grad_stats.items():
                history[k].append(v)

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

            epoch_time = time.perf_counter() - epoch_start
            sec_per_step = epoch_time / max(n_steps, 1)
            if self.verbose:
                summary = " ".join(f"{k}={v:.5g}" for k, v in
                                   {**running, **val_metrics, **grad_stats}.items())
                print(f"epoch {epoch}: {summary} "
                      f"time={epoch_time:.1f}s ({sec_per_step:.3f}s/step)")
            if self.logger:
                self.logger.log({"epoch": epoch, **running, **val_metrics, **grad_stats,
                                 "lr": self.optimizer.param_groups[0]["lr"],
                                 "epoch_time": epoch_time, "sec_per_step": sec_per_step})

            if self.checkpoint_interval and self.ckpt_path and (epoch + 1) % self.checkpoint_interval == 0:
                self._save(epoch, history)

        if self.logger:
            self.logger.close()
        self._save(self.epochs - 1, history)
        return dict(history)
