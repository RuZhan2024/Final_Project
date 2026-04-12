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
from functools import lru_cache


@lru_cache(maxsize=1)
def _dropout_types():
    try:
        import torch.nn as nn  # type: ignore
    except Exception:
        return ()
    return (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)


def _dropout_modules(model):
    cached = getattr(model, "_mc_dropout_modules", None)
    if cached is not None:
        return cached
    dtypes = _dropout_types()
    if not dtypes:
        return []
    mods = [
        m
        for m in model.modules()
        if isinstance(m, dtypes)
    ]
    try:
        setattr(model, "_mc_dropout_modules", mods)
    except Exception:
        pass
    return mods


def enable_dropout_only(model) -> None:
    """Enable dropout layers while keeping the model globally in eval mode."""
    for m in _dropout_modules(model):
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


@lru_cache(maxsize=16)
def _inv_n_table(max_n: int) -> tuple[float, ...]:
    n = int(max_n)
    if n < 1:
        return (0.0,)
    out = [0.0]
    out.extend((1.0 / float(i)) for i in range(1, n + 1))
    return tuple(out)


def mc_predict_mu_sigma(
    model,
    forward_fn: Callable[[], "Union[object, 'torch.Tensor']"],
    *,
    M: int = 20,
    return_samples: bool = False,
    max_sigma_for_early_stop: float | None = None,
    max_se_for_early_stop: float | None = None,
    min_M_for_early_stop: int = 4,
    return_n_used: bool = False,
) -> "Union[Tuple['torch.Tensor','torch.Tensor'], Tuple['torch.Tensor','torch.Tensor',int], Tuple['torch.Tensor','torch.Tensor','torch.Tensor'], Tuple['torch.Tensor','torch.Tensor','torch.Tensor',int]]":
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
    max_sigma_for_early_stop : float | None
        If provided (>0), stop sampling early once running sigma for all batch
        items is <= threshold and at least min_M_for_early_stop samples are used.
    max_se_for_early_stop : float | None
        Optional early-stop criterion on standard error (sigma / sqrt(N)).
        If provided (>0), the running standard error for all items must be <=
        threshold. If both sigma and SE thresholds are provided, both must hold.
    min_M_for_early_stop : int
        Minimum samples before early-stop can trigger.
    return_n_used : bool
        If True, also return the number of MC samples actually used.
    """
    import torch  # type: ignore

    M_i = int(M)
    M = M_i if M_i > 0 else 1
    sigma_stop = None
    if max_sigma_for_early_stop is not None:
        try:
            sigma_stop_v = float(max_sigma_for_early_stop)
            if sigma_stop_v > 0.0:
                sigma_stop = sigma_stop_v
        except Exception:
            sigma_stop = None
    se_stop = None
    if max_se_for_early_stop is not None:
        try:
            se_stop_v = float(max_se_for_early_stop)
            if se_stop_v > 0.0:
                se_stop = se_stop_v
        except Exception:
            se_stop = None
    min_M_i = int(min_M_for_early_stop)
    min_M_stop = min_M_i if min_M_i > 0 else 1
    has_early_stop = (sigma_stop is not None) or (se_stop is not None)
    need_stop_check = has_early_stop
    use_sigma_stop = sigma_stop is not None
    use_se_stop = se_stop is not None
    inv_n = None
    sigma_stop_sq = None
    se_stop_sq = None
    if use_sigma_stop:
        sigma_stop_sq = sigma_stop * sigma_stop
    if use_se_stop:
        se_stop_sq = se_stop * se_stop
        inv_n = _inv_n_table(M)
    inv_n_all = inv_n if inv_n is not None else _inv_n_table(M)
    was_training = bool(getattr(model, "training", False))
    fwd = forward_fn
    as1d = _as_1d
    # Start from a clean eval state.
    model.eval()
    dropout_mods = _dropout_modules(model)
    for m in dropout_mods:
        m.train()

    try:
        S = None
        with torch.inference_mode():
            if not dropout_mods:
                y = as1d(fwd())
                mu = y
                sigma = torch.zeros_like(y)
                if return_samples:
                    if return_n_used:
                        return mu, sigma, y.unsqueeze(0), 1
                    return mu, sigma, y.unsqueeze(0)
                if return_n_used:
                    return mu, sigma, 1
                return mu, sigma

            if M == 1:
                y = as1d(fwd())
                mu = y
                sigma = torch.zeros_like(y)
                if return_samples:
                    if return_n_used:
                        return mu, sigma, y.unsqueeze(0), 1
                    return mu, sigma, y.unsqueeze(0)
                if return_n_used:
                    return mu, sigma, 1
                return mu, sigma

            if return_samples:
                y0 = as1d(fwd())
                S = y0.new_empty((M, y0.shape[0]))
                S[0] = y0
                count = 1
                if not has_early_stop:
                    for i in range(1, M):
                        S[i] = as1d(fwd())
                        count += 1
                else:
                    mu_run = y0
                    m2_run = torch.zeros_like(y0)
                    if hasattr(m2_run, "arr"):
                        m2_max_fn = lambda x: float(x.arr.max())
                    elif hasattr(m2_run, "max"):
                        m2_max_fn = lambda x: float(x.max().item())
                    else:
                        m2_max_fn = lambda x: float(x)
                    for i in range(1, M):
                        y = as1d(fwd())
                        S[i] = y
                        count += 1
                        inv_count = inv_n_all[count]
                        delta = y - mu_run
                        mu_run = mu_run + (delta * inv_count)
                        delta2 = y - mu_run
                        m2_run = m2_run + (delta * delta2)
                        if count >= min_M_stop and need_stop_check:
                            m2_max = m2_max_fn(m2_run)
                            var_max = (m2_max if m2_max > 0.0 else 0.0) * inv_count
                            stop_ok = True
                            if use_sigma_stop and var_max > sigma_stop_sq:
                                stop_ok = False
                            if use_se_stop and (var_max * inv_count) > se_stop_sq:
                                stop_ok = False
                            if stop_ok:
                                break
                if count < M:
                    S = S[:count]
            else:
                y0 = as1d(fwd())
                mu_run = y0
                m2_run = torch.zeros_like(y0)
                if hasattr(m2_run, "arr"):
                    m2_max_fn = lambda x: float(x.arr.max())
                elif hasattr(m2_run, "max"):
                    m2_max_fn = lambda x: float(x.max().item())
                else:
                    m2_max_fn = lambda x: float(x)
                count = 1
                if not need_stop_check:
                    for _ in range(1, M):
                        y = as1d(fwd())
                        count += 1
                        inv_count = inv_n_all[count]
                        delta = y - mu_run
                        mu_run = mu_run + (delta * inv_count)
                        delta2 = y - mu_run
                        m2_run = m2_run + (delta * delta2)
                else:
                    for _ in range(1, M):
                        y = as1d(fwd())
                        count += 1
                        inv_count = inv_n_all[count]
                        delta = y - mu_run
                        mu_run = mu_run + (delta * inv_count)
                        delta2 = y - mu_run
                        m2_run = m2_run + (delta * delta2)
                        if count >= min_M_stop:
                            m2_max = m2_max_fn(m2_run)
                            var_max = (m2_max if m2_max > 0.0 else 0.0) * inv_count
                            stop_ok = True
                            if use_sigma_stop and var_max > sigma_stop_sq:
                                stop_ok = False
                            if use_se_stop and (var_max * inv_count) > se_stop_sq:
                                stop_ok = False
                            if stop_ok:
                                break
                # population variance (unbiased=False): var = M2 / N
                var = m2_run * inv_n_all[count]
                sigma = torch.sqrt(torch.clamp(var, min=0.0))
                if return_n_used:
                    return mu_run, sigma, int(count)
                return mu_run, sigma

        if S is None:
            raise ValueError("MC sampling produced no outputs")
        var_mean_fn = getattr(torch, "var_mean", None)
        if callable(var_mean_fn):
            var, mu = var_mean_fn(S, dim=0, unbiased=False)
            sigma = torch.sqrt(torch.clamp(var, min=0.0))
        else:
            mu = S.mean(dim=0)
            sigma = S.std(dim=0, unbiased=False)
        if return_samples:
            if return_n_used:
                return mu, sigma, S, int(S.shape[0])
            return mu, sigma, S
        if return_n_used:
            return mu, sigma, int(S.shape[0])
        return mu, sigma
    finally:
        # Always restore the original global mode, even if forward_fn fails.
        if was_training:
            model.train()
        else:
            model.eval()
