#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/uncertainty.py

MC Dropout utilities.

What MC Dropout is (in your project)
------------------------------------
Instead of running the model once, we run it M times with Dropout ON,
then summarize:
  mu    = mean(predictions)
  sigma = std(predictions)

In this repo we assume forward_fn returns probabilities in [0,1]
(i.e., sigmoid already applied), because alerting/triage thresholds
operate on probabilities, not logits.

IMPORTANT deployment behavior
-----------------------------
During MC sampling we want:
  ✅ Dropout layers ON   (to create stochastic outputs)
  ✅ BatchNorm OFF       (keep running stats stable; do NOT update BN buffers)

So the correct approach is:
  - set model.eval() globally  -> BN is eval, dropout becomes eval (OFF)
  - then re-enable ONLY dropout modules with .train()

That’s exactly what this file does.
"""

from __future__ import annotations

from typing import Callable, Tuple, Union, Optional

# NOTE: We import torch lazily inside functions so this file can be imported
# in contexts where torch may not be installed (e.g., lightweight tooling).
# In your training/eval runtime, torch will be present.


def enable_dropout_only(model) -> None:
    """
    Enable ONLY dropout modules inside the model.

    Why:
    - model.eval() disables dropout globally (good for BN, bad for MC sampling)
    - We want dropout active BUT BN frozen

    Implementation:
    - iterate over all submodules
    - if the module is a dropout layer, set it to train mode

    Important:
    - We do NOT call model.train() here, because that would also turn BN ON.
    """
    try:
        import torch.nn as nn  # type: ignore
    except Exception:
        return

    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
    )

    for m in model.modules():
        if isinstance(m, dropout_types):
            m.train()


def _as_1d(x):
    """
    Convert a forward output into a 1D tensor-like shape [B].

    forward_fn may return:
      - [B]                (already correct)
      - [B, 1]             (common for binary heads)
      - [B, K]             (we take the first column by convention)
      - scalar             (single item) -> shape [1]

    We keep this tolerant because different model heads sometimes differ.
    """
    if x is None:
        raise ValueError("forward_fn returned None")

    # Torch tensors have .ndim and .shape. This also works for numpy arrays.
    ndim = getattr(x, "ndim", 0)

    if ndim == 0:
        # scalar -> [1]
        return x.reshape(1)

    if ndim == 2 and x.shape[-1] == 1:
        # [B,1] -> [B]
        return x.squeeze(-1)

    if ndim >= 2:
        # [B,K,...] -> flatten everything but batch and take first channel
        return x.reshape(x.shape[0], -1)[:, 0]

    # ndim == 1 -> already [B]
    return x


def mc_predict_mu_sigma(
    model,
    forward_fn: Callable[[], "Union[object, 'torch.Tensor']"],
    *,
    M: int = 20,
    return_samples: bool = False,
) -> "Union[Tuple['torch.Tensor','torch.Tensor'], Tuple['torch.Tensor','torch.Tensor','torch.Tensor']]":
    """
    Monte-Carlo Dropout mean/std.

    Parameters
    ----------
    model : torch.nn.Module
        The model instance to sample.

    forward_fn : () -> torch.Tensor
        Callable that runs one forward pass and returns output for the
        CURRENT fixed inputs.

        In this project, forward_fn should return probabilities in [0,1],
        not logits. Example:
            forward_fn = lambda: torch.sigmoid(model(x))

    M : int
        Number of stochastic samples. Higher M => more stable sigma estimate but slower.

    return_samples : bool
        If True, also return the stacked samples tensor of shape [M, B].

    Returns
    -------
    (mu, sigma) or (mu, sigma, samples)

    mu    : torch.Tensor [B]
    sigma : torch.Tensor [B]
    samples (optional): torch.Tensor [M, B]
    """
    import torch  # type: ignore

    M = int(M) if int(M) > 0 else 1

    # Save original global mode so we can restore after MC sampling.
    was_training = bool(getattr(model, "training", False))

    samples = []

    with torch.no_grad():
        for _ in range(M):
            # Start every sample from eval state:
            # - BN eval (frozen)
            # - dropout eval (OFF)
            model.eval()

            # Then enable only dropout modules:
            # - dropout train (ON)
            # - BN remains eval (OFF)
            enable_dropout_only(model)

            # Run one stochastic forward pass
            y = forward_fn()

            # Ensure shape is [B]
            y = _as_1d(y)

            samples.append(y)

    # Stack M samples into [M, B]
    S = torch.stack(samples, dim=0)

    # Mean and std across the sampling dimension
    mu = S.mean(dim=0)
    sigma = S.std(dim=0, unbiased=False)  # unbiased=False is more stable for small M

    # Restore original global mode
    if was_training:
        model.train()
    else:
        model.eval()

    if return_samples:
        return mu, sigma, S
    return mu, sigma


# Backward-compatible alias (some codebases call it this)
mc_dropout_predict = mc_predict_mu_sigma
