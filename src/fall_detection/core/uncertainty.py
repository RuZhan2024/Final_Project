#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/uncertainty.py

MC Dropout utilities.

Important behaviour for deployment:
  - Keep BatchNorm (and other modules) in eval mode.
  - Enable *only* dropout layers during sampling.
  - Return (mu, sigma) in the same space as forward_fn outputs.
    (In this project we pass probability outputs, not logits.)
"""

from __future__ import annotations

from typing import Callable, Tuple, Union


def enable_dropout_only(model) -> None:
    """Enable dropout layers while keeping the model globally in eval mode."""
    try:
        import torch.nn as nn  # type: ignore
    except Exception:
        return

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()


def _as_1d(x):
    """Ensure output is [B] tensor."""
    # x might be [B], [B,1], or scalar.
    if x is None:
        raise ValueError("forward_fn returned None")
    if getattr(x, "ndim", 0) == 0:
        return x.reshape(1)
    if x.ndim == 2 and x.shape[-1] == 1:
        return x.squeeze(-1)
    if x.ndim >= 2:
        return x.reshape(x.shape[0], -1)[:, 0]
    return x


def mc_predict_mu_sigma(
    model,
    forward_fn: Callable[[], "Union[object, 'torch.Tensor']"],
    *,
    M: int = 20,
    return_samples: bool = False,
) -> "Union[Tuple['torch.Tensor','torch.Tensor'], Tuple['torch.Tensor','torch.Tensor','torch.Tensor']]":
    """Monte-Carlo Dropout mean/std.

    Parameters
    ----------
    model : torch.nn.Module
        Model to sample.
    forward_fn : () -> torch.Tensor
        Callable that returns model output for the current inputs.
        For deployment we expect *probabilities* in [0,1] (after sigmoid).
    M : int
        Number of MC samples.
    return_samples : bool
        If True, also return samples tensor of shape [M,B].
    """
    import torch  # type: ignore

    M = int(M) if int(M) > 0 else 1
    was_training = bool(getattr(model, "training", False))

    # Start from a clean eval state.
    model.eval()

    samples = []
    with torch.no_grad():
        for _ in range(M):
            enable_dropout_only(model)
            y = forward_fn()
            y = _as_1d(y)
            samples.append(y)

    S = torch.stack(samples, dim=0)  # [M,B]
    mu = S.mean(dim=0)
    sigma = S.std(dim=0, unbiased=False)

    # Restore original global mode.
    if was_training:
        model.train()
    else:
        model.eval()

    if return_samples:
        return mu, sigma, S
    return mu, sigma
