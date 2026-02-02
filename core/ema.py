#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/ema.py

Exponential Moving Average (EMA) of model parameters.

What EMA is (simple explanation)
--------------------------------
EMA keeps a "smoothed" version of your model weights over training steps.

If θ_t is the current parameter at step t, EMA keeps:
  ema_t = decay * ema_{t-1} + (1 - decay) * θ_t

Why it helps
------------
- Reduces noise from minibatch updates.
- Often improves validation stability and deployment robustness.
- Especially helpful when your validation metric is a bit "jumpy".

Important design choice in this repo
------------------------------------
We track ONLY *parameters* (model.named_parameters()).
We do NOT track BatchNorm buffers (running_mean/running_var).

Why:
- Buffers are not learnable parameters and are handled separately.
- In evaluation, you usually want to keep the model's real buffers
  and only swap the weights for EMA weights (more compatible).

Used by your training scripts
-----------------------------
Your trainers expect:
  ema = EMA(model, decay)
  ema.update(model)
  with ema.average_parameters(model):
      evaluate(model)
  ema.state_dict()
  ema.load_state_dict(...)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn


class EMA:
    """
    EMA tracker for a PyTorch model.

    Attributes
    ----------
    decay:
      EMA decay factor in (0, 1). Larger => smoother, slower updates.
      Common values: 0.99, 0.999, 0.9999

    shadow:
      Dict of parameter_name -> EMA tensor (same shape as parameter)

    _backup:
      Temporary backup of the model parameters used for swapping EMA weights in/out.
      Only used inside average_parameters() context.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        if not (0.0 <= self.decay < 1.0):
            # We allow decay=0 (EMA becomes "copy current weights").
            raise ValueError(f"EMA decay must be in [0,1). Got {self.decay}")

        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Optional[Dict[str, torch.Tensor]] = None

        # Initialize shadow weights from current model parameters.
        # We only track parameters that require gradients (trainable parameters).
        for name, p in model.named_parameters():
            if p.requires_grad:
                # detach() breaks the autograd graph; clone() makes a real copy.
                self.shadow[name] = p.detach().clone()

    def __repr__(self) -> str:
        return f"EMA(decay={self.decay}, tracked={len(self.shadow)})"

    # ------------------------------------------------------------
    # Internal helper: ensure shadow tensor lives on same device
    # ------------------------------------------------------------
    def _ensure_same_device(self, name: str, p: torch.Tensor) -> torch.Tensor:
        """
        Make sure the shadow tensor for this parameter is on the same device as p.

        Why:
        - If you create EMA before moving model to GPU/MPS, shadow tensors might be on CPU.
        - p.copy_(shadow) requires both tensors to be on the same device.
        """
        s = self.shadow[name]
        if s.device != p.device:
            s = s.to(p.device)
            self.shadow[name] = s
        return s

    # ------------------------------------------------------------
    # Update EMA with current model weights
    # ------------------------------------------------------------
    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update EMA (shadow) weights using current model weights.

        Math:
          shadow = decay * shadow + (1 - decay) * param

        Called once per optimizer step (typically).
        """
        d = float(self.decay)

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # If a parameter appears that wasn't in shadow (rare, but possible),
            # initialize it on the fly.
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
                continue

            s = self._ensure_same_device(name, p)

            # In-place update is faster and uses less memory.
            # mul_(d) then add_(p, alpha=1-d)
            s.mul_(d).add_(p.detach(), alpha=(1.0 - d))

    # ------------------------------------------------------------
    # Save / load EMA tensors
    # ------------------------------------------------------------
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Return a CPU-safe clone of EMA tensors.

        We clone to avoid callers accidentally modifying internal tensors.
        """
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """
        Load EMA tensors from a saved state dict.

        We clone so internal storage is independent from the caller.
        """
        state = state or {}
        self.shadow = {k: v.detach().clone() for k, v in state.items()}

    # ------------------------------------------------------------
    # Weight swapping utilities (used in evaluation)
    # ------------------------------------------------------------
    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """
        Backup current model parameters into self._backup.

        Called before copy_to() when entering average_parameters() context.

        Why we need a backup:
        - We temporarily overwrite model params with EMA params for evaluation.
        - After evaluation, we restore the original params to continue training.
        """
        self._backup = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """
        Copy EMA weights into model parameters.

        This is the "swap in EMA" step.
        """
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                continue

            s = self._ensure_same_device(name, p)
            p.copy_(s)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """
        Restore model parameters from backup.

        This is the "swap back original" step.
        """
        if not self._backup:
            # Nothing stored -> nothing to restore
            return

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self._backup:
                continue

            # Backup tensor is already detached and cloned.
            b = self._backup[name]
            if b.device != p.device:
                b = b.to(p.device)
            p.copy_(b)

        self._backup = None

    # ------------------------------------------------------------
    # Context manager: evaluate with EMA weights temporarily
    # ------------------------------------------------------------
    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """
        Temporarily swap model parameters to EMA weights.

        Usage:
          with ema.average_parameters(model):
              model.eval()
              ... run validation ...

        Implementation steps:
          1) store original params
          2) copy EMA params into model
          3) yield control to caller
          4) restore original params
        """
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)
