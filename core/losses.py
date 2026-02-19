#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/losses.py

Loss functions shared by trainers.

We keep this small and dependency-free so training scripts can import it
without introducing extra coupling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLogits(nn.Module):
    """Binary focal loss, operating directly on logits.

    This is a drop-in alternative to BCEWithLogitsLoss for imbalanced data.
    Typical values: alpha=0.25, gamma=2.0.

    Notes:
      - targets should be floats in {0,1} with shape [N] or [N,1].
      - logits can be any shape broadcast-compatible with targets.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of: mean|sum|none")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure float targets
        t = targets.float()
        # BCE per element
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * t + (1.0 - p) * (1.0 - t)  # p_t
        alpha_t = self.alpha * t + (1.0 - self.alpha) * (1.0 - t)
        loss = alpha_t * ((1.0 - pt).clamp(min=0.0) ** self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
