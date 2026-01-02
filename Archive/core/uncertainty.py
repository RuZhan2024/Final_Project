#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/uncertainty.py

MC Dropout utilities (Option 2).

We want epistemic uncertainty estimates without changing training:
- Keep the model in eval() mode for stable BatchNorm etc.
- Enable only Dropout modules during sampling.

Public API
----------
- mc_predict_mu_sigma(model, forward_fn, M) -> (mu, sigma)
  where forward_fn() returns a 1D torch tensor of probabilities/logits (caller chooses).

Notes
-----
MC Dropout is expensive on CPU; recommended usage:
- normal streaming: M=1 (no uncertainty)
- confirm window only: M=8..16
"""

from __future__ import annotations

from typing import Callable, Tuple, Optional
import torch
import torch.nn as nn


def enable_dropout_only(model: nn.Module) -> None:
    """Set Dropout modules to train() while leaving the overall model in eval()."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()


@torch.no_grad()
def mc_predict_mu_sigma(
    model: nn.Module,
    forward_fn: Callable[[], torch.Tensor],
    M: int = 12,
    return_samples: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Run MC Dropout sampling.

    Parameters
    ----------
    model : nn.Module
        Model (should be in eval() mode).
    forward_fn : Callable[[], torch.Tensor]
        Closure that runs a forward pass and returns a 1D tensor (shape [B] or []).
        The returned tensor should already be probabilities in [0,1] if you want mu/sigma on probs.
    M : int
        Number of stochastic forward passes.
    return_samples : bool
        If True, also returns the samples tensor of shape [M, ...].

    Returns
    -------
    mu : torch.Tensor
    sigma : torch.Tensor
    samples : Optional[torch.Tensor]
    """
    if M <= 1:
        s = forward_fn()
        mu = s
        sigma = torch.zeros_like(s)
        return mu, sigma, s.unsqueeze(0) if return_samples else None

    model.eval()
    enable_dropout_only(model)

    samples = []
    for _ in range(int(M)):
        s = forward_fn()
        samples.append(s.detach().float())

    S = torch.stack(samples, dim=0)  # [M, ...]
    mu = S.mean(dim=0)
    sigma = S.std(dim=0, unbiased=False)

    if return_samples:
        return mu, sigma, S
    return mu, sigma, None
