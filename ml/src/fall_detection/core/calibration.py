#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/calibration.py

Temperature scaling for binary logits.

Contract C (repo-wide): calibration is **temperature scaling only**.
We calibrate a scalar T>0 on validation logits, then apply:
  p = sigmoid(logits / T)

We store T and summary statistics in YAML, and consistently apply it in:
  train → fit OPs → evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid used by calibration reporting helpers."""
    x = np.asarray(x, dtype=np.float64)
    # stable sigmoid
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out.astype(np.float32)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Scale binary logits by a validated temperature parameter."""
    T = float(T)
    if not np.isfinite(T) or T <= 0:
        raise ValueError("Temperature T must be finite and >0")
    return (np.asarray(logits, dtype=np.float32) / T).astype(np.float32)


def nll_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    """Compute binary NLL directly from logits and 0/1 targets."""
    lt = torch.tensor(np.asarray(logits, dtype=np.float32)).view(-1)
    yt = torch.tensor(np.asarray(y, dtype=np.float32)).view(-1)
    loss = F.binary_cross_entropy_with_logits(lt, yt)
    return float(loss.detach().cpu().item())


def ece_from_probs(probs: np.ndarray, y: np.ndarray, *, n_bins: int = 15) -> float:
    """Expected calibration error for binary probabilities."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    yt = np.asarray(y, dtype=np.int32).reshape(-1)
    if p.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        acc = float(yt[m].mean())
        conf = float(p[m].mean())
        w = float(m.mean())
        ece += w * abs(acc - conf)
    return float(ece)


@dataclass
class TempFitResult:
    """Temperature-fit summary persisted alongside a calibrated checkpoint.

    The before/after statistics are part of the artifact contract: downstream
    tooling can report whether calibration helped without having to rerun the
    optimization step.
    """
    T: float
    n_val: int
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float

    def to_yaml_dict(self) -> Dict[str, object]:
        """Serialize the calibration artifact in the repo's YAML schema."""
        return {
            "calibration": {
                "method": "temperature",
                "T": float(self.T),
                "n_val": int(self.n_val),
                "nll_before": float(self.nll_before),
                "nll_after": float(self.nll_after),
                "ece_before": float(self.ece_before),
                "ece_after": float(self.ece_after),
            }
        }


def fit_temperature(
    logits: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 75,
    lr: float = 0.05,
    init_T: float = 1.0,
) -> TempFitResult:
    """Fit a scalar temperature T on validation logits using NLL minimisation.

    Only labels in {0,1} participate in the fit. Invalid labels are discarded
    so mixed evaluation arrays can still reuse this helper safely.
    """
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    m = (y == 0) | (y == 1)
    logits = logits[m]
    y = y[m]
    if logits.size == 0:
        return TempFitResult(T=1.0, n_val=0, nll_before=float("nan"), nll_after=float("nan"), ece_before=float("nan"), ece_after=float("nan"))

    # Convert once to torch because optimization updates only the scalar
    # temperature parameter, not the stored validation logits/labels.
    lt = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y.astype(np.float32), dtype=torch.float32)

    # Softplus keeps the learned temperature positive without manual clipping,
    # which would otherwise distort gradients near the boundary.
    t_param = torch.nn.Parameter(torch.tensor(float(init_T), dtype=torch.float32))
    opt = torch.optim.Adam([t_param], lr=float(lr))

    def T() -> torch.Tensor:
        return F.softplus(t_param) + 1e-6

    for _ in range(int(max_iter)):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(lt / T(), yt)
        loss.backward()
        opt.step()

    T_hat = float(T().detach().cpu().item())
    nll_before = nll_from_logits(logits, y)
    nll_after = nll_from_logits(logits / max(1e-6, T_hat), y)
    ece_before = ece_from_probs(sigmoid(logits), y)
    ece_after = ece_from_probs(sigmoid(logits / max(1e-6, T_hat)), y)

    return TempFitResult(
        T=float(T_hat),
        n_val=int(len(y)),
        nll_before=float(nll_before),
        nll_after=float(nll_after),
        ece_before=float(ece_before),
        ece_after=float(ece_after),
    )
