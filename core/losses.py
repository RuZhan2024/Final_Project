#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/losses.py

Loss functions used across trainers.

This repo mainly uses:
- BCEWithLogitsLoss for TCN training
- BinaryFocalLossWithLogits for GCN training (often more robust to imbalance)

Why "with logits"?
------------------
Most binary heads output *logits* (unbounded real numbers).
We apply sigmoid internally in the loss function for numerical stability.

Key term: pos_weight
--------------------
PyTorch BCEWithLogitsLoss supports a `pos_weight` tensor:
- It multiplies the loss contribution of positive examples (y=1).
- A common choice is: pos_weight = n_neg / max(n_pos, 1)

This is different from focal alpha (see below):
- pos_weight is a direct reweighting inside BCE
- focal alpha is a balance factor in focal loss
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _squeeze_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor to shape [B].

    Many training loops produce:
      logits: [B] or [B,1]
      targets: [B] or [B,1]

    This helper makes both cases consistent.
    """
    if x.ndim == 2 and x.shape[-1] == 1:
        return x.view(-1)
    if x.ndim == 1:
        return x
    # If someone passes weird shapes, try a safe flatten-by-batch:
    return x.reshape(x.shape[0], -1)[:, 0]


def make_pos_weight_from_counts(n_pos: int, n_neg: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Build the `pos_weight` tensor for BCEWithLogitsLoss.

    Recommended formula:
      pos_weight = n_neg / max(n_pos, 1)

    Interpretation:
    - If positives are rare, n_neg/n_pos is large => positives get up-weighted.
    - If dataset is balanced, pos_weight ~ 1.

    Returns:
      torch.Tensor shape [1] (PyTorch expects this shape for binary tasks)
    """
    n_pos = int(n_pos)
    n_neg = int(n_neg)
    w = float(n_neg) / float(max(1, n_pos))
    return torch.tensor([w], device=device, dtype=dtype)


class BinaryFocalLossWithLogits(nn.Module):
    """
    Binary focal loss operating on logits.

    Focal loss idea (intuitive)
    ---------------------------
    BCE already penalizes wrong predictions. Focal loss adds a factor that:
    - down-weights EASY examples (already predicted correctly with high confidence)
    - focuses the loss on HARD examples (misclassified or uncertain)

    Args
    ----
    alpha:
      Balance factor in [0,1].
      - If alpha > 0.5, positives get more emphasis.
      - If alpha < 0.5, negatives get more emphasis.
      Common choice: 0.25 (from original focal loss paper).

    gamma:
      Focusing parameter (>=0).
      - gamma = 0 reduces to alpha-balanced BCE
      - larger gamma focuses more on hard examples (common: 2.0)

    pos_weight:
      Optional tensor like BCEWithLogitsLoss.pos_weight (shape [1]).
      This multiplies the BCE term for positives before focal scaling.
      You typically use EITHER pos_weight OR alpha balancing.
      (But using both is allowed if you intentionally want stronger imbalance handling.)

    reduction:
      "mean" | "sum" | "none"

    ignore_index:
      If set (e.g., -1), targets equal to ignore_index are excluded from loss.
      Useful if you ever mix unlabeled windows (y=-1) into a batch by mistake.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        ignore_index: Optional[float] = None,
    ) -> None:
        super().__init__()

        # ---- validate inputs early (fail fast = easier debugging) ----
        self.alpha = float(alpha)
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0,1], got {self.alpha}")

        self.gamma = float(gamma)
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {self.gamma}")

        reduction = str(reduction).lower().strip()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be mean|sum|none, got {reduction}")
        self.reduction = reduction

        # pos_weight should be tensor shape [1] for binary BCEWithLogitsLoss
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        logits:
          [B] or [B,1] unbounded real values

        targets:
          [B] or [B,1] labels (0/1). Can be int or float.
        """
        # 1) Normalize shapes to [B]
        logits = _squeeze_1d(logits)
        t = _squeeze_1d(targets).float()

        # 2) Optional ignore mask (e.g., ignore_index=-1)
        if self.ignore_index is not None:
            valid = (t != float(self.ignore_index))
            if not torch.any(valid):
                # If everything is ignored, return 0 so training doesn't NaN.
                return logits.sum() * 0.0
            logits = logits[valid]
            t = t[valid]

        # 3) Targets should be in {0,1}. We clamp to be safe if small noise exists.
        t = t.clamp(0.0, 1.0)

        # 4) BCE per-example (no reduction yet)
        #
        # BCEWithLogitsLoss for each example:
        #   bce_i = - [ t*log(sigmoid(logit)) + (1-t)*log(1-sigmoid(logit)) ]
        #
        # We use F.binary_cross_entropy_with_logits which is stable.
        bce = F.binary_cross_entropy_with_logits(
            logits,
            t,
            reduction="none",
            pos_weight=self.pos_weight,
        )

        # 5) pt = probability of the TRUE class
        #
        # p  = sigmoid(logits) = P(y=1)
        # pt = p if t=1 else (1-p)
        p = torch.sigmoid(logits)
        pt = p * t + (1.0 - p) * (1.0 - t)

        # 6) alpha_t balances class contributions
        #
        # alpha_t = alpha if t=1 else (1-alpha)
        alpha_t = self.alpha * t + (1.0 - self.alpha) * (1.0 - t)

        # 7) focal scaling factor: (1-pt)^gamma
        #
        # - If pt is high (easy example), (1-pt)^gamma is small => down-weight loss
        # - If pt is low (hard example), factor is large => focus loss
        focal_factor = torch.pow((1.0 - pt).clamp(0.0, 1.0), self.gamma)

        # 8) Final per-example focal loss
        loss = alpha_t * focal_factor * bce

        # 9) Reduction
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
