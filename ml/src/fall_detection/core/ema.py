#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/ema.py

Lightweight exponential moving average (EMA) of model parameters.

Used to stabilise training + evaluation. This module is intentionally small
and dependency-free besides torch.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator

import torch


@dataclass
class EMAState:
    decay: float
    shadow: Dict[str, torch.Tensor]


class EMA:
    """EMA wrapper for a torch.nn.Module.

    Typical usage:
        ema = EMA(model, decay=0.999)
        ... after each opt.step(): ema.update(model)
        with ema.apply(model):   # temporarily swap to EMA params
            evaluate(...)
    """

    def __init__(self, model: torch.nn.Module, *, decay: float = 0.999) -> None:
        if not (0.0 < float(decay) < 1.0):
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self._init_from(model)

    def _init_from(self, model: torch.nn.Module) -> None:
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def update(self, model: torch.nn.Module) -> None:
        """Update EMA shadow weights from the current model weights.

        Only floating tensors are EMA-updated. Integer/bool tensors are copied.
        This avoids failures with buffers like BatchNorm.num_batches_tracked.
        """
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k not in self.shadow:
                    self.shadow[k] = v.detach().clone()
                    continue

                # For integer/bool buffers, copy directly.
                if not torch.is_floating_point(v):
                    self.shadow[k] = v.detach().clone()
                    continue

                sv = self.shadow[k]
                if not torch.is_floating_point(sv):
                    sv = v.detach().clone()

                if sv.device != v.device or sv.dtype != v.dtype:
                    sv = sv.to(device=v.device, dtype=v.dtype)

                sv.mul_(self.decay).add_(v.detach(), alpha=(1.0 - self.decay))
                self.shadow[k] = sv

    @contextmanager
    def apply(self, model: torch.nn.Module) -> Iterator[None]:
        """Temporarily load EMA weights into the model, then restore."""
        # backup
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        try:
            model.load_state_dict(self.shadow, strict=False)
            yield
        finally:
            model.load_state_dict(self._backup, strict=False)
            self._backup = {}

    def use(self, model: torch.nn.Module):
        """Alias for apply(), kept for backward compatibility with training scripts.

        Usage:
            with ema.use(model):
                evaluate(...)
        """
        return self.apply(model)

        """Alias for apply(), kept for trainer compatibility."""
        return self.apply(model)

    def state_dict(self) -> Dict[str, object]:
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", {})
        if isinstance(shadow, dict):
            self.shadow = {str(k): v.detach().clone() for k, v in shadow.items() if isinstance(v, torch.Tensor)}
